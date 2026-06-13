from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def load_markdown_documents(docs_dir: Path) -> list[tuple[Path, str]]:
    docs: list[tuple[Path, str]] = []
    for path in sorted(docs_dir.glob("*.md")):
        docs.append((path, path.read_text(encoding="utf-8")))
    return docs


def _title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_documents(
    docs: list[tuple[Path, str]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path, text in docs:
        title = _title_from_markdown(text, path.stem.replace("-", " ").title())
        clean = re.sub(r"\n{3,}", "\n\n", text).strip()
        start = 0
        index = 0
        while start < len(clean):
            end = min(start + chunk_size, len(clean))
            if end < len(clean):
                boundary = clean.rfind("\n\n", start, end)
                if boundary > start + 300:
                    end = boundary
            chunk_text = clean[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{path.stem}-{index}",
                        text=chunk_text,
                        metadata={
                            "source": str(path),
                            "title": title,
                            "chunk_id": f"{path.stem}-{index}",
                        },
                    )
                )
            index += 1
            start = max(end - overlap, end) if end == len(clean) else max(end - overlap, start + 1)
    return chunks

