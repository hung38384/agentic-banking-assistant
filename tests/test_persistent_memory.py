from app.services.memory import ConversationMemory


def test_memory_does_not_persist_without_consent(tmp_path):
    path = tmp_path / "memory.json"
    memory = ConversationMemory(persist_path=path, persist_requires_consent=True)

    memory.set_consent("conv-1", {"consent_confirmed": False})
    memory.update_preferences("conv-1", "I prefer Vietnamese and credit card information")
    memory.add("conv-1", "user", "hello")

    assert not path.exists() or "conv-1" not in path.read_text(encoding="utf-8")
    assert memory.context("conv-1")["persistent_memory_allowed"] is False


def test_memory_persists_and_reloads_with_consent(tmp_path):
    path = tmp_path / "memory.json"
    memory = ConversationMemory(persist_path=path, persist_requires_consent=True)

    memory.set_consent("conv-2", {"consent_confirmed": True})
    memory.update_preferences("conv-2", "Tôi muốn hỏi về thẻ tín dụng")
    memory.update_slots("conv-2", {"loan_amount": 100000000})
    memory.add("conv-2", "user", "hello")

    reloaded = ConversationMemory(persist_path=path, persist_requires_consent=True)
    context = reloaded.context("conv-2")

    assert context["persistent_memory_allowed"] is True
    assert context["preferences"]["product_interest"] == "credit_card"
    assert context["slots"]["loan_amount"] == 100000000
    assert context["history"][0]["content"] == "hello"
