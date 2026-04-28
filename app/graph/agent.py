"""
CropEye Agent powered by Groq (llama-3.1-8b-instant).
- Uses Llama 3.1 8B Instant via Groq API as the LLM
- Keeps in-process chat history per (session_id, plot_id)
- Calls backend API tools based on user intent
- Multilingual: auto-detects and responds in Hindi/Marathi/Kannada/English
"""
import re
from dataclasses import dataclass

from groq import AsyncGroq
from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.knowledge.app_knowledge import CROPEYE_APP_KNOWLEDGE
from app.tools.api_tools import (
    get_growth_map,
    get_nutrient_analysis,
    get_pest_map,
    get_plot_info,
    get_soil_moisture,
    get_soil_moisture_map,
    get_water_uptake_map,
    get_weather,
)

SYSTEM_PROMPT = f"""You are **CropEye Assistant** — a smart, friendly AI chatbot for Indian farmers using the CropEye precision farming app.

## Your Role
You help farmers:
1. Understand and use every feature of the CropEye mobile app
2. Analyse their field data (soil, water, pests, nutrients, growth)
3. Get actionable farming recommendations based on real satellite data
4. Navigate the app easily

## Language Rules (CRITICAL)
- Detect the language of each user message automatically
- Reply in the SAME language the user wrote in
- Supported: English, Hindi (हिंदी), Marathi (मराठी), Kannada (ಕನ್ನಡ)
- If mixing languages (Hinglish), match their style
- Use simple, clear language farmers will understand
- For numbers/values, always include units (%, kg/ha, °C, mm)

## App Knowledge
{CROPEYE_APP_KNOWLEDGE}

## When to Use Tools
- User asks about water / irrigation / water map → call get_water_uptake_map
  → Returns: deficient%, less%, adequate%, excellent%, excess% pixel distribution
- User asks about crop health / NDVI / vegetation / growth map → call get_growth_map
  → Returns: healthy%, moderate%, weak%, stress% pixel distribution
- User asks about pests / pest map / fungal / chewing / sucking → call get_pest_map
  → Returns: chewing%, fungi%, sucking%, wilt%, SoilBorn% affected pixels
- User asks about soil moisture trend / history / last 7 days → call get_soil_moisture
  → Returns: daily moisture%, rainfall mm, ET mm for each of the last 7 days
- User asks about soil moisture map / soil zones / moisture distribution → call get_soil_moisture_map
  → Returns: less%, adequate%, excellent%, excess%, shallow_water% pixel distribution
- User asks about NPK / fertilizer / nutrients → call get_nutrient_analysis
- User asks about weather / temperature → call get_weather (need lat/lon from context)
- User asks about field details → call get_plot_info

## Response Style
- Be conversational and warm — talk like a helpful farming expert friend
- For data analysis: key finding → details → recommendation
- Always end data responses with a practical action
- Use emojis occasionally 🌾💧🌡️

## AI Recommendations Format
1. **Current Status**: what the data shows
2. **Risk/Concern**: what this means for the crop
3. **Action**: what to do right now
4. **Timeline**: when to do it

## Important Notes
- Plot ID is provided with each message — always use it for API calls
- Prices in the app come from AGMARKNET (government market data)
- Weather data is real-time from OpenWeatherMap
- Satellite data updates every few days (Sentinel-2 revisit cycle)
"""


@dataclass
class _StateSnapshot:
    values: dict


def _extract_user_question(full_message: str) -> str:
    marker = "User question:"
    if marker in full_message:
        return full_message.split(marker, 1)[1].strip()
    return full_message.strip()


def _extract_context_value(full_message: str, label: str) -> str:
    pattern = rf"{label}:\s*([^\|\n]+)"
    match = re.search(pattern, full_message)
    return match.group(1).strip() if match else ""


def _extract_lat_lon(full_message: str) -> tuple[float | None, float | None]:
    match = re.search(r"lat=([-+]?\d*\.?\d+),\s*lon=([-+]?\d*\.?\d+)", full_message)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


class CropEyeAgent:
    def __init__(self) -> None:
        self._history: dict[str, list] = {}
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def _run_tools(self, user_question: str, full_message: str, plot_id: str) -> tuple[list[str], list[str]]:
        q = user_question.lower()
        called_tools: list[str] = []
        tool_results: list[str] = []

        def should_call(words: list[str]) -> bool:
            return any(w in q for w in words)

        # ── Soil Moisture MAP (pixel classification) ──────────────────────
        # Check this BEFORE the broader water/irrigation check to avoid double-calling
        if should_call(["soil moisture map", "soil map", "moisture map",
                        "moisture distribution", "moisture zone", "soil zone",
                        "how much field is dry", "how much field is wet"]):
            called_tools.append("get_soil_moisture_map")
            tool_results.append(await get_soil_moisture_map.ainvoke({"plot_id": plot_id}))

        # ── Water Uptake MAP (irrigation depth) ───────────────────────────
        elif should_call(["water", "irrig", "ndwi", "water uptake", "water map",
                          "field water", "is my field dry", "do i need to water"]):
            called_tools.append("get_water_uptake_map")
            tool_results.append(await get_water_uptake_map.ainvoke({"plot_id": plot_id}))

        # ── Growth MAP (NDVI / crop health) ───────────────────────────────
        if should_call(["ndvi", "growth", "vegetation", "crop health", "healthy crop",
                        "growth map", "crop stress", "how are my crops", "field health"]):
            called_tools.append("get_growth_map")
            tool_results.append(await get_growth_map.ainvoke({"plot_id": plot_id}))

        # ── Pest Detection MAP ─────────────────────────────────────────────
        if should_call(["pest", "insect", "bug", "aphid", "whitefly", "fungal",
                        "fungi", "chewing", "sucking", "wilt", "pest map", "pest risk"]):
            called_tools.append("get_pest_map")
            tool_results.append(await get_pest_map.ainvoke({"plot_id": plot_id}))

        # ── Soil Moisture TREND (7-day history) ───────────────────────────
        if should_call(["soil moisture", "moisture history", "last week moisture",
                        "7 day moisture", "moisture trend", "moisture level",
                        "soil water level", "moisture graph"]):
            if "get_soil_moisture_map" not in called_tools:
                called_tools.append("get_soil_moisture")
                tool_results.append(await get_soil_moisture.ainvoke({"plot_id": plot_id}))

        # ── NPK / Nutrients ────────────────────────────────────────────────
        if should_call(["npk", "nitrogen", "phosphorus", "potassium", "fertilizer",
                        "nutrient", "urea", "dap", "soil fertility", "soil nutrient"]):
            plantation_date = _extract_context_value(full_message, "Plantation date")
            called_tools.append("get_nutrient_analysis")
            tool_results.append(
                await get_nutrient_analysis.ainvoke(
                    {"plot_id": plot_id, "plantation_date": plantation_date}
                )
            )

        # ── Weather ───────────────────────────────────────────────────────
        if should_call(["weather", "temperature", "rain", "wind", "humidity", "forecast",
                        "spray today", "going to rain"]):
            lat, lon = _extract_lat_lon(full_message)
            if lat is not None and lon is not None:
                called_tools.append("get_weather")
                tool_results.append(await get_weather.ainvoke({"lat": lat, "lon": lon}))

        # ── Plot / Field Info ──────────────────────────────────────────────
        if should_call(["plot", "field details", "field info", "my field",
                        "plantation date", "field area", "crop name"]):
            called_tools.append("get_plot_info")
            tool_results.append(await get_plot_info.ainvoke({"plot_id": plot_id, "access_token": ""}))

        return called_tools, tool_results

    async def ainvoke(self, payload: dict, config: dict) -> dict:
        messages = payload.get("messages", [])
        if not messages:
            return {"messages": []}
        human_message = messages[-1]
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        plot_id = thread_id.split("::", 1)[1] if "::" in thread_id else ""

        history = self._history.setdefault(thread_id, [])
        history.append(HumanMessage(content=human_message.content))

        user_question = _extract_user_question(human_message.content)
        called_tools, tool_outputs = await self._run_tools(user_question, human_message.content, plot_id)

        prior_turns = []
        for m in history[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            prior_turns.append(f"{role}: {m.content}")
        convo_text = "\n".join(prior_turns[-10:])
        tool_text = "\n\n".join(tool_outputs) if tool_outputs else "No external tool used."

        user_content = (
            f"Conversation so far:\n{convo_text or 'No prior messages.'}\n\n"
            f"Current user question:\n{user_question}\n\n"
            f"Relevant tool outputs:\n{tool_text}\n"
        )

        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or "Sorry, I could not process your request."
        ai_msg = AIMessage(content=answer, tool_calls=[{"name": t, "args": {}, "id": t, "type": "tool_call"} for t in called_tools])
        history.append(ai_msg)
        return {"messages": history}

    def get_state(self, config: dict) -> _StateSnapshot:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return _StateSnapshot(values={"messages": self._history.get(thread_id, [])})

    def update_state(self, config: dict, data: dict) -> None:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        self._history[thread_id] = data.get("messages", [])


_agent: CropEyeAgent | None = None


def get_agent() -> CropEyeAgent:
    global _agent
    if _agent is None:
        _agent = CropEyeAgent()
    return _agent


def get_thread_config(session_id: str, plot_id: str) -> dict:
    """
    Each (session_id, plot_id) pair gets its own isolated memory thread.
    """
    return {"configurable": {"thread_id": f"{session_id}::{plot_id}"}}
