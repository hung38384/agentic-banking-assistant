import argparse
import asyncio
import json
import os
import sys
import time
import types
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import BankingAgent
from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.vector_store import BankingVectorStore


DEFAULT_CASES = [
    {
        "id": "r001",
        "question": "What is the credit card annual fee waiver condition?",
        "reference": "The annual fee is 299,000 VND and can be waived when annual card spending reaches 60,000,000 VND.",
    },
    {
        "id": "r002",
        "question": "What is the minimum opening balance for the MB Digital Savings Account?",
        "reference": "The minimum opening balance is 50,000 VND.",
    },
    {
        "id": "r003",
        "question": "When is the home loan early repayment fee waived?",
        "reference": "Early repayment within the first 3 years may incur a 1% to 3% fee; after year 3 this fee may be waived according to product policy.",
    },
    {
        "id": "r004",
        "question": "Can you tell me the current deposit interest rate for the savings account?",
        "reference": "The assistant must not invent deposit rates if the current rate table is not available.",
    },
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run real RAGAS evaluation against the banking agent.")
    parser.add_argument("--cases", type=int, default=4, help="Number of cases to run.")
    parser.add_argument("--app-provider", default=os.getenv("RAGAS_APP_PROVIDER", "mock"))
    parser.add_argument("--judge-provider", default=os.getenv("RAGAS_JUDGE_PROVIDER", "groq"))
    parser.add_argument("--skip-answer-relevancy", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only build and print the RAGAS dataset preview.")
    parser.add_argument("--timeout", type=int, default=45, help="RAGAS judge timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=1, help="RAGAS judge retry count.")
    parser.add_argument(
        "--mode",
        choices=["metric", "evaluate"],
        default="metric",
        help="metric uses RAGAS DiscreteMetric.score; evaluate uses ragas.evaluate built-ins.",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.semantic_cache_enabled = False
    settings.llm_provider = args.app_provider
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    agent = BankingAgent(settings, store, ConversationMemory())

    samples = []
    start = time.perf_counter()
    for item in DEFAULT_CASES[: args.cases]:
        state = await agent.run(f"ragas-{item['id']}", item["question"], {})
        contexts = [chunk.get("text", "") for chunk in state.get("retrieved_chunks", [])]
        samples.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": state.get("answer", ""),
                "contexts": contexts,
                "ground_truth": item["reference"],
                "user_input": item["question"],
                "response": state.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": item["reference"],
            }
        )

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "cases": len(samples),
                    "sample_columns": sorted(samples[0].keys()) if samples else [],
                    "preview": samples,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    try:
        ragas_result = _run_ragas(
            samples,
            settings,
            args.judge_provider,
            args.skip_answer_relevancy,
            args.timeout,
            args.max_retries,
            args.mode,
        )
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": f"Missing optional RAGAS dependency: {exc.name}",
                    "install": 'pip install -e ".[ragas]"',
                    "prepared_cases": len(samples),
                },
                indent=2,
            )
        )
        return
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": str(exc),
                    "prepared_cases": len(samples),
                },
                indent=2,
            )
        )
        return

    print(
        json.dumps(
            {
                "status": "completed",
                "judge_provider": args.judge_provider,
                "cases": len(samples),
                "suite_latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "metrics": ragas_result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _run_ragas(
    samples: list[dict[str, Any]],
    settings: Any,
    judge_provider: str,
    skip_answer_relevancy: bool,
    timeout: int,
    max_retries: int,
    mode: str,
) -> dict[str, Any]:
    _install_ragas_compat_shims()
    datasets = import_module("datasets")
    ragas = import_module("ragas")
    ragas_metrics = import_module("ragas.metrics")
    ragas_run_config = import_module("ragas.run_config")

    dataset = datasets.Dataset.from_list(samples)
    llm, embeddings = _build_ragas_models(settings, judge_provider)
    if mode == "metric":
        return _run_ragas_discrete_metrics(samples, ragas_metrics, llm, include_answer_relevancy=not skip_answer_relevancy)

    metrics = _build_metrics(ragas_metrics, include_answer_relevancy=not skip_answer_relevancy and embeddings is not None)

    result = ragas.evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=ragas_run_config.RunConfig(timeout=timeout, max_retries=max_retries),
        raise_exceptions=False,
    )
    return _result_to_dict(result)


def _run_ragas_discrete_metrics(
    samples: list[dict[str, Any]],
    ragas_metrics: Any,
    llm: Any,
    include_answer_relevancy: bool,
) -> dict[str, Any]:
    DiscreteMetric = getattr(ragas_metrics, "DiscreteMetric")
    metric_defs = {
        "faithfulness": DiscreteMetric(
            name="faithfulness",
            allowed_values=["pass", "fail"],
            prompt=(
                "You are evaluating a banking RAG answer. Return only pass or fail.\n"
                "Pass if every factual claim and number in the response is supported by the retrieved contexts "
                "or the response correctly refuses unsupported information.\n"
                "User question: {user_input}\nRetrieved contexts: {retrieved_contexts}\nResponse: {response}"
            ),
        ),
        "context_precision": DiscreteMetric(
            name="context_precision",
            allowed_values=["pass", "fail"],
            prompt=(
                "You are evaluating retrieved banking contexts. Return only pass or fail.\n"
                "Pass if the retrieved contexts are relevant to answering the user question and do not mostly "
                "contain unrelated banking information.\n"
                "User question: {user_input}\nRetrieved contexts: {retrieved_contexts}"
            ),
        ),
    }
    if include_answer_relevancy:
        metric_defs["answer_relevancy"] = DiscreteMetric(
            name="answer_relevancy",
            allowed_values=["pass", "fail"],
            prompt=(
                "You are evaluating answer relevancy. Return only pass or fail.\n"
                "Pass if the response directly addresses the user question or clearly refuses when the "
                "requested information is unsupported.\n"
                "User question: {user_input}\nReference answer: {reference}\nResponse: {response}"
            ),
        )

    rows = []
    for sample in samples:
        row = {"id": sample["id"]}
        for name, metric in metric_defs.items():
            result = metric.score(
                llm=llm,
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts="\n\n".join(sample["retrieved_contexts"]),
                reference=sample["reference"],
            )
            row[name] = _metric_value(result)
        rows.append(row)

    summary = {}
    for name in metric_defs:
        values = [1.0 if row[name] == "pass" else 0.0 for row in rows]
        summary[name] = round(sum(values) / max(len(values), 1), 4)
    summary["rows"] = rows
    return summary


def _install_ragas_compat_shims() -> None:
    """RAGAS 0.4.x imports an old VertexAI module even when VertexAI is unused."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used to satisfy optional import
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def _build_ragas_models(settings: Any, judge_provider: str) -> tuple[Any, Any | None]:
    langchain_openai = import_module("langchain_openai")
    ragas_llms = import_module("ragas.llms")
    ragas_embeddings = import_module("ragas.embeddings")
    openai_module = import_module("openai")

    ChatOpenAI = getattr(langchain_openai, "ChatOpenAI")
    OpenAIEmbeddings = getattr(langchain_openai, "OpenAIEmbeddings")
    LangchainLLMWrapper = getattr(ragas_llms, "LangchainLLMWrapper")
    llm_factory = getattr(ragas_llms, "llm_factory", None)
    LangchainEmbeddingsWrapper = getattr(ragas_embeddings, "LangchainEmbeddingsWrapper")
    OpenAI = getattr(openai_module, "OpenAI")

    provider = judge_provider.lower()
    if provider == "groq":
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for RAGAS judge provider 'groq'.")
        if llm_factory is not None:
            judge_llm = llm_factory(
                settings.groq_model,
                client=OpenAI(api_key=api_key, base_url=settings.groq_base_url),
            )
        else:
            judge_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=settings.groq_model,
                    api_key=api_key,
                    base_url=settings.groq_base_url,
                    temperature=0,
                )
            )
    elif provider == "openai":
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for RAGAS judge provider 'openai'.")
        model = os.getenv("RAGAS_OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        if llm_factory is not None:
            judge_llm = llm_factory(model, client=OpenAI(api_key=api_key))
        else:
            judge_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=0,
                )
            )
    else:
        raise RuntimeError("Unsupported RAGAS judge provider. Use 'groq' or 'openai'.")

    embeddings = None
    openai_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key:
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=os.getenv("RAGAS_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=openai_key,
            )
        )
    return judge_llm, embeddings


def _build_metrics(ragas_metrics: Any, include_answer_relevancy: bool) -> list[Any]:
    metric_names = ["faithfulness", "context_precision"]
    if include_answer_relevancy:
        metric_names.append("answer_relevancy")

    metrics = []
    for name in metric_names:
        metric = getattr(ragas_metrics, name, None)
        if metric is not None:
            metrics.append(metric)
            continue
        class_name = {
            "faithfulness": "Faithfulness",
            "context_precision": "LLMContextPrecisionWithoutReference",
            "answer_relevancy": "ResponseRelevancy",
        }[name]
        metric_class = getattr(ragas_metrics, class_name, None)
        if metric_class is not None:
            metrics.append(metric_class())
    if not metrics:
        raise RuntimeError("No compatible RAGAS metrics found in installed ragas version.")
    return metrics


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        numeric = frame.select_dtypes(include="number")
        return {column: round(float(numeric[column].mean()), 4) for column in numeric.columns}
    if hasattr(result, "to_dict"):
        raw = result.to_dict()
    elif isinstance(result, dict):
        raw = result
    else:
        raw = dict(result)
    return {
        key: round(float(value), 4) if isinstance(value, int | float) else value
        for key, value in raw.items()
        if not str(key).startswith("_")
    }


def _metric_value(result: Any) -> Any:
    if hasattr(result, "value"):
        return result.value
    if hasattr(result, "to_dict"):
        data = result.to_dict()
        return data.get("value", data)
    return result


if __name__ == "__main__":
    asyncio.run(main())
