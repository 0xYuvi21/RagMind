"""
Test doubles built on LangChain's own fake primitives, wrapped behind the same
LLMProvider protocol (Backend/interfaces.py) the real GroqLLMProvider
implements — tests exercise the exact composition path production uses
(Backend/deps.py -> Retrieve/llmquery.build_chat_agent), just with a scripted
model and hash-based embeddings instead of live Groq/HuggingFace calls.
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult


class ScriptedFakeChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel doesn't implement bind_tools (raises
    NotImplementedError), which create_react_agent calls unconditionally.
    Overriding it as a no-op keeps the model returning its scripted responses
    regardless of which tools were bound.

    Also records every list of messages it's asked to generate from into
    `capture` (if set) — this is what test_agent_memory.py uses to assert a
    follow-up question's LLM input actually contains the prior turn, without
    depending on the scripted answer's content."""

    # Typed as Any (not Optional[list]) so pydantic passes the object through
    # by reference instead of re-validating/copying it into a new list —
    # otherwise appends here wouldn't be visible on the caller's original list.
    capture: Any = None

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.capture is not None:
            self.capture.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_tool_calling_response(tool_name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": tool_name, "args": args, "id": call_id}])


def make_final_answer(text: str) -> AIMessage:
    return AIMessage(content=text)


class FakeLLMProvider:
    """LLMProvider protocol implementation for tests. `responses` cycles in
    order across however many model calls a single agent.invoke() makes
    (typically: one tool-calling AIMessage, then one final-answer AIMessage).
    A fresh model instance (i=0) is built on every get_chat_model() call —
    matching production, where build_chat_agent() is called fresh per
    ConversationService.ask() — but all instances append to the same shared
    `capture` list if one is passed in, so a test can inspect every model
    call across multiple /chat turns in the same conversation."""

    def __init__(self, responses: List[BaseMessage], capture: Optional[list] = None):
        self._responses = responses
        self._capture = capture

    def get_chat_model(self) -> ScriptedFakeChatModel:
        return ScriptedFakeChatModel(responses=self._responses, capture=self._capture)


def make_fake_embeddings() -> DeterministicFakeEmbedding:
    """Hash-based, deterministic, no network/model download — fine for
    isolation tests where what matters is that chat B's store never returns
    chat A's vectors, not semantic retrieval quality."""
    return DeterministicFakeEmbedding(size=64)
