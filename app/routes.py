"""
CropEye Chatbot Routes
POST   /api/v1/chat         — main chat endpoint
GET    /api/v1/chat/history  — get chat history for a session
DELETE /api/v1/chat/history  — clear chat history for a session
"""
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from app.models import ChatRequest, ChatResponse, ClearHistoryRequest
from app.graph.agent import get_agent, get_thread_config

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    - plot_id: farm field identifier
    - user_message: any language (English/Hindi/Marathi/Kannada)
    - session_id: unique per user (phone number or UUID)
    """
    try:
        agent = get_agent()
        config = get_thread_config(req.session_id, req.plot_id)

        # Inject field context before the user message
        context_prefix = _build_context(req)
        full_message = f"{context_prefix}\n\nUser question: {req.user_message}"

        # Invoke LangGraph ReAct agent
        # MemorySaver automatically loads + saves thread history
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=full_message)]},
            config=config,
        )

        # Extract last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_messages[-1].content if ai_messages else "Sorry, I could not process your request."

        tool_calls = _extract_tool_calls(result["messages"])

        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            plot_id=req.plot_id,
            data_sources_used=tool_calls,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")


@router.get("/chat/history")
async def get_history(session_id: str, plot_id: str):
    """Returns chat history for a session from LangGraph MemorySaver."""
    try:
        agent = get_agent()
        config = get_thread_config(session_id, plot_id)

        # MemorySaver exposes state via get_state
        state = agent.get_state(config)
        messages = state.values.get("messages", []) if state.values else []

        history = []
        for m in messages:
            if isinstance(m, HumanMessage):
                content = m.content
                if "User question:" in content:
                    content = content.split("User question:")[-1].strip()
                history.append({"role": "user", "content": content})
            elif isinstance(m, AIMessage):
                history.append({"role": "assistant", "content": m.content})

        return {
            "session_id": session_id,
            "plot_id": plot_id,
            "message_count": len(history),
            "messages": history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/history")
async def clear_history(req: ClearHistoryRequest):
    """Clears in-memory chat history for a specific session + plot thread."""
    try:
        agent = get_agent()
        config = get_thread_config(req.session_id, req.plot_id)

        # Update state to empty messages — clears the thread
        agent.update_state(config, {"messages": []})

        return {
            "status": "cleared",
            "session_id": req.session_id,
            "plot_id": req.plot_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────
def _build_context(req: ChatRequest) -> str:
    parts = [f"[CONTEXT] Plot ID: {req.plot_id}"]
    if req.farmer_name:
        parts.append(f"Farmer: {req.farmer_name}")
    if req.crop_type:
        parts.append(f"Crop: {req.crop_type}")
    if req.plantation_date:
        parts.append(f"Plantation date: {req.plantation_date}")
    if req.lat and req.lon:
        parts.append(f"Field GPS: lat={req.lat}, lon={req.lon}")
    return " | ".join(parts)


def _extract_tool_calls(messages) -> list[str]:
    tool_names = set()
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_names.add(tc.get("name", ""))
    return list(tool_names)
