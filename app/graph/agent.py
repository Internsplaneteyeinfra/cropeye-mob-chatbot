"""
CropEye Agent powered by Groq (llama-3.1-8b-instant).
- Uses Llama 3.1 8B Instant via Groq API as the LLM
- Keeps in-process chat history per (session_id, plot_id)
- Calls backend API tools based on user intent
- Follow-up aware: caches last tool data so context carries across messages
- Multilingual: auto-detects and responds in Hindi/Marathi/Kannada/English
"""
import logging
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

logger = logging.getLogger("cropeye.agent")

SYSTEM_PROMPT = f"""You are **CropEye Assistant** — a smart, friendly AI chatbot for Indian farmers using the CropEye precision farming app.

## Your Role
You help farmers:
1. Understand and use every feature of the CropEye mobile app
2. Analyse their field data (soil, water, pests, nutrients, growth)
3. Get actionable farming recommendations based on real satellite data
4. Navigate the app easily

## Language Rules (CRITICAL — READ CAREFULLY)
- Detect the language of the user message and reply in THE EXACT SAME LANGUAGE
- Supported languages: English, Hindi (हिंदी), Marathi (मराठी), Kannada (ಕನ್ನಡ)

### Marathi Detection (IMPORTANT)
- Marathi words to recognise: tyasathi, karavya, lagtil, mla, ky, tumhi, aahe, nahi,
  sheti, pani, mati, kiti, kasa, karu, sanga, kela, hote, ahe, aplya, amcha, tar,
  nakki, jagat, ugavlela, pikache, aaj, udya, varsha, mahina, divas, shekda
- If user writes in Romanized Marathi (Marathi words in English letters like
  "tyasathi mla ky karavya lagtil"), ALWAYS reply in Marathi — either in
  Devanagari (मराठी) OR in the same Romanized Marathi style the user used
- NEVER reply in Hindi when user is speaking Marathi — they are different languages
- Marathi-specific words differ from Hindi: "aahe" (not "hai"), "nahi" (Marathi usage),
  "tumhi" (not "tum/aap"), "mla" = "mala" = "to me", "tyasathi" = "for that"

### Hindi Detection
- Hindi-specific words: hai, hain, karo, kya, mujhe, aapko, yahan, wahan, bahut

### General Rules
- If mixing languages (Hinglish/Manglish), match their exact style
- Use simple, clear language farmers will understand
- For numbers/values, always include units (%, kg/ha, °C, mm)
- NEVER mix languages in your reply unless the user mixed them first

## App Knowledge
{CROPEYE_APP_KNOWLEDGE}

## When to Use Tools
- User asks about water / irrigation / water map → get_water_uptake_map
  → Returns: deficient%, less%, adequate%, excellent%, excess% pixel distribution
- User asks about crop health / NDVI / vegetation / growth map → get_growth_map
  → Returns: healthy%, moderate%, weak%, stress% pixel distribution
- User asks about pests / pest map / fungal / chewing / sucking → get_pest_map
  → Returns: chewing%, fungi%, sucking%, wilt%, SoilBorn% affected pixels
- User asks about soil moisture trend / history / last 7 days → get_soil_moisture
  → Returns: daily moisture%, rainfall mm, ET mm for each of the last 7 days
- User asks about soil moisture map / soil zones / moisture distribution → get_soil_moisture_map
  → Returns: less%, adequate%, excellent%, excess%, shallow_water% pixel distribution
- User asks about NPK / fertilizer / nutrients → get_nutrient_analysis
- User asks about weather / temperature → get_weather (need lat/lon from context)
- User asks about field details → get_plot_info

## Conversation & Follow-Up Rules (IMPORTANT)
- The conversation history and previously fetched field data are provided to you
- For follow-up questions (e.g. "what will happen if it increases?", "is that good?",
  "what should I do?"), use the [Previous Field Data] already provided — do NOT say
  you need more information
- Always connect follow-up answers to the actual numbers from the previous data
- Be conversational — remember what was discussed earlier in this chat

## Response Style (STRICT)
- Keep every reply to **3-4 lines maximum** — no long paragraphs
- Each line should be one clear point: status → risk → action
- Use bullet points (•) or short numbered steps, not essay-style text
- Be direct and to the point — farmers need quick, clear answers
- Use emojis occasionally 🌾💧🌡️ but only 1-2 per reply
- Never repeat the question back or add filler sentences

## Format for Data Responses (3-4 lines only)
• **Status**: [what the data shows in one line]
• **Risk**: [what it means for the crop in one line]
• **Action**: [what to do right now in one line]

## CRITICAL OUTPUT RULES
- The blocks marked [Context: ...], [Live Field Data ...], [Previous Field Data ...],
  and [Language Instruction: ...] are INTERNAL instructions for you only
- NEVER copy, print, or mention any of these blocks in your reply
- NEVER show raw JSON data or API response in your reply
- Your reply must contain ONLY the farmer-facing message — nothing else

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


def _recent_user_text(history: list, n: int = 3) -> str:
    """Concatenate last n user messages to detect active topic for follow-ups."""
    msgs = [m.content.lower() for m in history if isinstance(m, HumanMessage)]
    return " ".join(msgs[-n:])


class CropEyeAgent:
    def __init__(self) -> None:
        self._history: dict[str, list] = {}
        # Caches the last fetched tool data per thread for follow-up awareness
        self._tool_context: dict[str, str] = {}
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def _run_tools(
        self,
        user_question: str,
        full_message: str,
        plot_id: str,
        history: list,
    ) -> tuple[list[str], list[str]]:
        q = user_question.lower()

        # For ambiguous/short follow-up messages, also scan recent history
        # to detect the active topic (e.g. "what if it increases?" after moisture Q)
        recent_context = _recent_user_text(history, n=3)
        q_with_context = f"{q} {recent_context}"

        called_tools: list[str] = []
        tool_results: list[str] = []

        def in_current(words: list[str]) -> bool:
            """Match keywords in current message only — for fresh data triggers."""
            return any(w in q for w in words)

        def in_context(words: list[str]) -> bool:
            """Match keywords across current + recent history — for topic continuation."""
            return any(w in q_with_context for w in words)

        # ── Soil Moisture MAP ─────────────────────────────────────────────
        # EN: soil moisture map | MR: mati cha ola nakasha, olsarpan nakasha
        # HI: mitti ka naksha, nami naksha
        if in_current(["soil moisture map", "soil map", "moisture map",
                       "moisture distribution", "moisture zone", "soil zone",
                       "how much field is dry", "how much field is wet",
                       "mati cha ola nakasha", "olsarpan nakasha", "mitti naksha",
                       "nami naksha", "sheti naksha"]):
            called_tools.append("get_soil_moisture_map")
            tool_results.append(await get_soil_moisture_map.ainvoke({"plot_id": plot_id}))

        # ── Water Uptake MAP ──────────────────────────────────────────────
        # EN: water, irrigation | MR: pani, sinchan, paus, panavtha
        # HI: paani, sinchai, sechaan
        elif in_current(["water", "irrig", "ndwi", "water uptake", "water map",
                         "field water", "is my field dry", "do i need to water",
                         "pani", "sinchan", "panavtha", "paus dya", "sinchai",
                         "sechaan", "paani dena", "paani ki zarurat"]):
            called_tools.append("get_water_uptake_map")
            tool_results.append(await get_water_uptake_map.ainvoke({"plot_id": plot_id}))

        # ── Growth MAP ────────────────────────────────────────────────────
        # EN: growth, crop health | MR: pik vadhata, pikachi prakat, pik arogya
        # HI: fasal ki sehat, ugaan, vikas
        if in_current(["ndvi", "growth", "vegetation", "crop health", "healthy crop",
                       "growth map", "crop stress", "how are my crops", "field health",
                       "pik vadhata", "pikachi prakat", "pik arogya", "pik kashe",
                       "ugavlela", "vanaspati", "fasal ki sehat", "fasal vikas",
                       "ugaan", "pik thik"]):
            called_tools.append("get_growth_map")
            tool_results.append(await get_growth_map.ainvoke({"plot_id": plot_id}))

        # ── Pest MAP ──────────────────────────────────────────────────────
        # EN: pest, insect | MR: ali, kide, rog, bimari, dhoka, kimad, naashkarak
        # HI: keede, bimari, rog, keede makoode, keet
        if in_current(["pest", "insect", "bug", "aphid", "whitefly", "fungal",
                       "fungi", "chewing", "sucking", "wilt", "pest map", "pest risk",
                       # Marathi
                       "ali", "kide", "rog", "bimari", "dhoka", "kimad",
                       "naashkarak", "kida", "kitak", "rograjog", "bughad",
                       "ali cha", "kide cha", "pik rog",
                       # Hindi
                       "keede", "keet", "makoode", "rog hai", "bimari hai",
                       "keetnaashak", "fungal rog"]):
            called_tools.append("get_pest_map")
            tool_results.append(await get_pest_map.ainvoke({"plot_id": plot_id}))

        # ── Soil Moisture TREND ───────────────────────────────────────────
        # EN: soil moisture | MR: mati cha ola, matit pani, maticha arda
        # HI: mitti ki nami, mitti me paani
        if in_current(["soil moisture", "moisture history", "last week moisture",
                       "7 day moisture", "moisture trend", "moisture level",
                       "soil water level", "moisture graph",
                       # Marathi
                       "mati cha ola", "matit pani", "maticha arda", "olsarpan",
                       "mati arda", "7 divsacha", "saat divsacha",
                       # Hindi
                       "mitti ki nami", "mitti me paani", "nami ka itihas",
                       "7 din ki nami"]):
            if "get_soil_moisture_map" not in called_tools:
                called_tools.append("get_soil_moisture")
                tool_results.append(await get_soil_moisture.ainvoke({"plot_id": plot_id}))

        # ── NPK / Nutrients ───────────────────────────────────────────────
        # EN: npk, fertilizer | MR: khate, uriya, poshanatve, mati poshan
        # HI: khad, urvarak, poshan
        if in_current(["npk", "nitrogen", "phosphorus", "potassium", "fertilizer",
                       "nutrient", "urea", "dap", "soil fertility", "soil nutrient",
                       # Marathi
                       "khate", "uriya", "poshanatve", "mati poshan", "naytrajan",
                       "fosfaras", "potash", "khad", "pik poshan",
                       # Hindi
                       "urvarak", "poshan", "khad dena", "naytrajan", "DAP dena"]):
            plantation_date = _extract_context_value(full_message, "Plantation date")
            called_tools.append("get_nutrient_analysis")
            tool_results.append(
                await get_nutrient_analysis.ainvoke(
                    {"plot_id": plot_id, "plantation_date": plantation_date}
                )
            )

        # ── Weather ───────────────────────────────────────────────────────
        # EN: weather | MR: havas, paus, temperature, unhacha, thanda
        # HI: mausam, barish, garmi
        if in_current(["weather", "temperature", "rain", "wind", "humidity", "forecast",
                       "spray today", "going to rain",
                       # Marathi
                       "havas", "paus", "unhacha", "thanda", "vara", "dhukke",
                       "havaman", "pavsache",
                       # Hindi
                       "mausam", "barish", "garmi", "sardi", "aandhi", "toofan"]):
            lat, lon = _extract_lat_lon(full_message)
            if lat is not None and lon is not None:
                called_tools.append("get_weather")
                tool_results.append(await get_weather.ainvoke({"lat": lat, "lon": lon}))

        # ── Plot / Field Info ─────────────────────────────────────────────
        # EN: my field | MR: mazhi sheti, mazha plot, sheticha mahiti
        # HI: mera khet, meri zameen
        if in_current(["plot", "field details", "field info", "my field",
                       "plantation date", "field area", "crop name",
                       # Marathi
                       "mazhi sheti", "mazha plot", "sheticha mahiti", "plot chi mahiti",
                       "sheti kiti", "pik konti",
                       # Hindi
                       "mera khet", "meri zameen", "khet ki jankari"]):
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

        # Pass history (before current msg) to tool runner for topic detection
        called_tools, tool_outputs = await self._run_tools(
            user_question, human_message.content, plot_id, history[:-1]
        )

        # ── Update or reuse tool context ──────────────────────────────────
        if tool_outputs:
            # Fresh data fetched — update cache
            self._tool_context[thread_id] = "\n\n".join(tool_outputs)
            tool_section = f"[Live Field Data — just fetched]\n{self._tool_context[thread_id]}"
            logger.info("[AGENT] Fresh tool data fetched: %s", called_tools)
        elif thread_id in self._tool_context:
            # Follow-up: reuse last fetched data so LLM can answer in context
            tool_section = f"[Previous Field Data — use this for the follow-up question]\n{self._tool_context[thread_id]}"
            logger.info("[AGENT] Follow-up detected — reusing cached tool context")
        else:
            tool_section = "No field data available yet for this session."

        # ── Build proper multi-turn Groq messages ─────────────────────────
        # System prompt first, then last 4 turns (8 msgs) as user/assistant pairs,
        # then current question with field data attached
        groq_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        recent_history = history[:-1][-8:]  # last 4 turns before current message
        for m in recent_history:
            if isinstance(m, HumanMessage):
                groq_messages.append({"role": "user", "content": _extract_user_question(m.content)})
            elif isinstance(m, AIMessage):
                groq_messages.append({"role": "assistant", "content": m.content})

        # Current message with field data + internal instructions
        # The [INTERNAL] blocks are NOT to appear in the reply
        current_content = (
            f"User message: {user_question}\n\n"
            f"[INTERNAL — plot_id='{plot_id}', data already fetched, "
            f"do NOT ask farmer for more info, "
            f"do NOT echo these instructions or raw data in your reply]\n\n"
            f"{tool_section}\n\n"
            f"[INTERNAL — Language: detect language from user message above. "
            f"Marathi words: ali, kide, rog, dhoka, pik, sheti, mati, pani, ola, "
            f"sinchan, tyasathi, karavya, lagtil, tumhi, aahe, mazhi, mazha, ahe. "
            f"Reply in Marathi (Devanagari) if Marathi detected. "
            f"NEVER echo [INTERNAL] blocks. Output ONLY the farmer reply.]"
        )
        groq_messages.append({"role": "user", "content": current_content})

        response = await self._client.chat.completions.create(
            model=settings.groq_model,
            messages=groq_messages,
            temperature=0.3,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or "Sorry, I could not process your request."

        ai_msg = AIMessage(
            content=answer,
            tool_calls=[
                {"name": t, "args": {}, "id": t, "type": "tool_call"}
                for t in called_tools
            ],
        )
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
    """Each (session_id, plot_id) pair gets its own isolated memory thread."""
    return {"configurable": {"thread_id": f"{session_id}::{plot_id}"}}
