"""
POST /chat/admin
POST /chat/admin/clear

Conversational endpoint for the ops-room admin. Backed by ChatAgent,
which can query live site status, search emergency protocols, pull the
daily report, or dispatch a real operational action if the admin asks
for one explicitly. Conversation history is kept in-memory per
session_id (core.chat_store) — not persisted across server restarts.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from agents.chat_agent import ChatAgent
from core import chat_store

router = APIRouter(prefix="/chat", tags=["chat"])
_chat_agent = ChatAgent()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    actions_taken: list[dict]


@router.post("/admin", response_model=ChatResponse)
def chat_admin(request: ChatRequest) -> ChatResponse:
    history = chat_store.get_history(request.session_id)
    # Tool-call/tool-result messages are internal to one turn's ReAct loop —
    # replaying them on the next turn adds noise without much benefit, so we
    # keep only plain user/assistant turns in the persisted history.
    plain_history = [m for m in history if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]

    outcome = _chat_agent.run(plain_history, request.message)
    for msg in outcome["new_messages"]:
        chat_store.append(request.session_id, msg)

    return ChatResponse(reply=outcome["reply"], actions_taken=outcome["actions_taken"])


@router.post("/admin/clear")
def chat_admin_clear(session_id: str = "default") -> dict:
    chat_store.clear(session_id)
    return {"cleared": True, "session_id": session_id}
