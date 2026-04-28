"""
CropEye LangGraph Tools — each tool calls a real backend API
and returns structured data that the LLM can interpret.
"""
import httpx
import logging
from langchain_core.tools import tool
from app.config import settings
import json
from datetime import date as _date

logger = logging.getLogger("cropeye.tools")

# ── shared async HTTP client ───────────────────────────────────────────────
_client = httpx.AsyncClient(timeout=45.0)

SAR_BASE = settings.plot_layer_base_url


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 1 — Water Uptake Map (Irrigation Depth Analysis)
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_water_uptake_map(plot_id: str) -> str:
    """
    Fetches water uptake / irrigation depth analysis for a farm field.
    Returns pixel-level moisture data: deficient%, less%, adequate%, excellent%, excess%.
    Use when user asks about: irrigation, water, moisture, field water status,
    do I need to water, is my field dry, water uptake, NDWI, water map.
    Endpoint: GET /wateruptake  (SAR Index Mapping API)
    """
    try:
        today = _date.today().isoformat()
        url = f"{SAR_BASE}/wateruptake"
        logger.info("[API CALL] get_water_uptake_map → GET %s plot=%s", url, plot_id)
        resp = await _client.get(url, params={"plot_name": plot_id, "end_date": today})
        if resp.status_code == 405:
            resp = await _client.post(url, params={"plot_name": plot_id, "end_date": today})
        data = resp.json()

        ps = data.get("pixel_summary", {})
        if not ps:
            return "Water uptake data not available for this plot."

        result = {
            "analysis": "Water Uptake / Irrigation Map Analysis",
            "plot_id": plot_id,
            "total_pixels_analysed": ps.get("total_pixel_count", 0),
            "pixel_summary": {
                "deficient_pixel_percentage": ps.get("deficient_pixel_percentage", 0),
                "less_pixel_percentage": ps.get("less_pixel_percentage", 0),
                "adequat_pixel_percentage": ps.get("adequat_pixel_percentage", 0),
                "excellent_pixel_percentage": ps.get("excellent_pixel_percentage", 0),
                "excess_pixel_percentage": ps.get("excess_pixel_percentage", 0),
            },
            "sensor": ps.get("sensor_used", "Sentinel-1/2"),
            "latest_image_date": ps.get("latest_image_date", ""),
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching water data: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 2 — Growth Map (NDVI / Crop Health)
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_growth_map(plot_id: str) -> str:
    """
    Fetches NDVI (Normalized Difference Vegetation Index) crop growth analysis.
    Returns vegetation health pixel data: healthy%, moderate%, weak%, stress%.
    Use when user asks about: crop health, vegetation, NDVI, growth, plants growing well,
    crop stress, field health, how are my crops, are crops healthy, growth map.
    Endpoint: GET /analyze_Growth  (SAR Index Mapping API)
    """
    try:
        today = _date.today().isoformat()
        url = f"{SAR_BASE}/analyze_Growth"
        logger.info("[API CALL] get_growth_map → GET %s plot=%s", url, plot_id)
        resp = await _client.get(url, params={"plot_name": plot_id, "end_date": today})
        if resp.status_code == 405:
            resp = await _client.post(url, params={"plot_name": plot_id, "end_date": today})
        data = resp.json()

        features = data.get("features", [])
        props = features[0].get("properties", {}) if features else {}
        ps = data.get("pixel_summary", {})

        result = {
            "analysis": "NDVI Crop Growth Map Analysis",
            "plot_id": plot_id,
            "data_source": props.get("data_source", "Sentinel-2"),
            "latest_image_date": props.get("letest_image_date") or props.get("latest_image_date"),
            "analysis_period": {
                "start": props.get("start_date"),
                "end": props.get("end_date"),
            },
            "pixel_summary": {
                "healthy_pixel_percentage": ps.get("healthy_pixel_percentage", 0),
                "moderate_pixel_percentage": ps.get("moderate_pixel_percentage", 0),
                "weak_pixel_percentage": ps.get("weak_pixel_percentage", 0),
                "stress_pixel_percentage": ps.get("stress_pixel_percentage", 0),
            },
            "tile_url_available": bool(props.get("tile_url")),
            "note": "healthy>60% = good crop stand. stress/weak>30% = urgent attention needed.",
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching growth data: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 3 — Pest Detection Map
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_pest_map(plot_id: str) -> str:
    """
    Fetches pest detection satellite map analysis for a farm field.
    Returns pest-type pixel percentages: chewing%, fungi%, sucking%, wilt%, soil-born%.
    Use when user asks about: pests, insects, bugs, aphids, whitefly, fungal disease,
    pest risk, pest attack, pest spray, are there pests, pest forecast, pest map.
    Endpoint: GET /pest-detection  (SAR Index Mapping API)
    """
    try:
        today = _date.today().isoformat()
        url = f"{SAR_BASE}/pest-detection"
        logger.info("[API CALL] get_pest_map → GET %s plot=%s", url, plot_id)
        resp = await _client.get(url, params={"plot_name": plot_id, "end_date": today})
        if resp.status_code == 405:
            resp = await _client.post(url, params={"plot_name": plot_id, "end_date": today})
        data = resp.json()

        features = data.get("features", [])
        props = features[0].get("properties", {}) if features else {}
        ps = data.get("pixel_summary", {})

        result = {
            "analysis": "Pest Detection Map Analysis",
            "plot_id": plot_id,
            "latest_image_date": props.get("letest_image_date") or props.get("latest_image_date"),
            "sensor": props.get("sensor_used", "Sentinel-1 SAR"),
            "pixel_summary": {
                "chewing_affected_pixel_percentage": ps.get("chewing_affected_pixel_percentage", 0),
                "fungi_affected_pixel_percentage": ps.get("fungi_affected_pixel_percentage", 0),
                "sucking_affected_pixel_percentage": ps.get("sucking_affected_pixel_percentage", 0),
                "wilt_affected_pixel_percentage": ps.get("wilt_affected_pixel_percentage", 0),
                "SoilBorn_affected_pixel_percentage": ps.get("SoilBorn_affected_pixel_percentage", 0),
            },
            "note": "Any category >10% warrants scouting. >30% = urgent spray action needed.",
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching pest data: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4 — Soil Moisture Trend (7-day historical time-series)
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_soil_moisture(plot_id: str) -> str:
    """
    Fetches 7-day soil moisture trend history from SAR Index Mapping satellite data.
    Returns daily moisture %, rainfall mm, and evapotranspiration for each day.
    Use when user asks about: soil moisture trend, soil moisture history, moisture graph,
    soil moisture level, last 7 days moisture, how was moisture this week.
    Endpoint: GET /soil-moisture/{plot_name}  (SAR Index Mapping API)
    """
    try:
        url = f"{SAR_BASE}/soil-moisture/{plot_id}"
        logger.info("[API CALL] get_soil_moisture → GET %s", url)
        resp = await _client.get(url, headers={"Accept": "application/json"})
        data = resp.json()

        stack = data.get("soil_moisture_stack", [])
        if not stack:
            return "No soil moisture trend data available for this plot."

        last7 = stack[-7:]
        avg = sum(d["soil_moisture"] for d in last7) / len(last7)

        if avg < 40:
            status = "Low — irrigation needed urgently"
        elif avg <= 80:
            status = "Good (optimal range 40–80%)"
        else:
            status = "High — risk of waterlogging"

        result = {
            "analysis": "7-Day Soil Moisture Trend",
            "plot_id": data.get("plot_name", plot_id),
            "average_moisture_pct": round(avg, 2),
            "status": status,
            "optimal_range": "40–80%",
            "daily_data": [
                {
                    "date": d["day"],
                    "moisture_pct": round(d["soil_moisture"], 2),
                    "rainfall_mm": round(d.get("rainfall_mm_yesterday", 0), 2),
                    "evapotranspiration_mm": round(d.get("et_mean_mm_yesterday", 0), 2),
                }
                for d in last7
            ],
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching soil moisture trend: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4b — Soil Moisture Map (SAR pixel classification)
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_soil_moisture_map(plot_id: str) -> str:
    """
    Fetches the SAR-based soil moisture MAP for a farm field.
    Returns pixel-level moisture classification: less%, adequate%, excellent%, excess%, shallow_water%.
    Use when user asks about: soil moisture map, soil map, moisture distribution,
    how much of the field is dry/wet, soil moisture zones, field moisture status.
    Endpoint: POST /SoilMoisture  (SAR Index Mapping API)
    """
    try:
        today = _date.today().isoformat()
        url = f"{SAR_BASE}/SoilMoisture"
        logger.info("[API CALL] get_soil_moisture_map → GET %s plot=%s", url, plot_id)
        resp = await _client.get(url, params={"plot_name": plot_id, "end_date": today})
        if resp.status_code == 405:
            resp = await _client.post(url, params={"plot_name": plot_id, "end_date": today})
        data = resp.json()

        ps = data.get("pixel_summary", {})
        if not ps:
            return "Soil moisture map data not available for this plot."

        features = data.get("features", [])
        props = features[0].get("properties", {}) if features else {}

        result = {
            "analysis": "Soil Moisture Map (SAR Pixel Classification)",
            "plot_id": plot_id,
            "sensor": ps.get("sensor_used", "Sentinel-1 SAR"),
            "latest_image_date": ps.get("latest_image_date") or props.get("latest_image_date", ""),
            "analysis_period": {
                "start": ps.get("start_date", ""),
                "end": ps.get("end_date", ""),
            },
            "total_pixels": ps.get("total_pixel_count", 0),
            "pixel_summary": {
                "less_pixel_percentage": ps.get("less_pixel_percentage", 0),
                "adequate_pixel_percentage": ps.get("adequate_pixel_percentage", 0),
                "excellent_pixel_percentage": ps.get("excellent_pixel_percentage", 0),
                "excess_pixel_percentage": ps.get("excess_pixel_percentage", 0),
                "shallow_water_pixel_percentage": ps.get("shallow_water_pixel_percentage", 0),
            },
            "note": "less>50% = irrigation needed. excess>20% = drainage risk. excellent+adequate = healthy.",
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching soil moisture map: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 5 — NPK / Nutrient Analysis
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_nutrient_analysis(plot_id: str, plantation_date: str = "") -> str:
    """
    Fetches NPK (Nitrogen, Phosphorus, Potassium) soil nutrient analysis.
    Use when user asks about: nitrogen, phosphorus, potassium, NPK, fertilizer,
    nutrients, soil nutrients, urea, DAP, soil fertility, should I apply fertilizer.
    """
    try:
        today = _date.today().isoformat()
        p_date = plantation_date or "2025-01-01"

        # Fetch NPK
        url_npk = f"{settings.soil_param_api_url}/required-n/{plot_id}"
        logger.info("[API CALL] get_nutrient_analysis NPK → POST %s plot=%s", url_npk, plot_id)
        resp_npk = await _client.post(
            url_npk,
            params={"plantation_date": p_date, "end_date": today},
            headers={"accept": "application/json"}
        )
        npk = resp_npk.json() if resp_npk.status_code == 200 else {}

        # Fetch pH, CEC, OC
        url_analysis = f"{settings.soil_param_api_url}/analyze-npk/{plot_id}"
        logger.info("[API CALL] get_nutrient_analysis pH/CEC/OC → POST %s plot=%s", url_analysis, plot_id)
        resp_analysis = await _client.post(
            url_analysis,
            params={"plantation_date": p_date, "date": today, "fe_days_back": 30},
            headers={"accept": "application/json"}
        )
        analysis = resp_analysis.json() if resp_analysis.status_code == 200 else {}
        ss = analysis.get("soil_statistics", analysis)

        nitrogen = npk.get("soilN", 0)
        phosphorus = npk.get("soilP", 0)
        potassium = npk.get("soilK", 0)
        ph = ss.get("phh2o", 7.0)
        cec = ss.get("cation_exchange_capacity", 0)
        oc = ss.get("organic_carbon_stock", 0)

        # Simple recommendations
        recs = []
        if nitrogen < 100:
            recs.append("Apply Urea (46% N) — nitrogen is deficient")
        if phosphorus < 30:
            recs.append("Apply DAP — phosphorus is low")
        if potassium < 60:
            recs.append("Apply MOP (potash) — potassium needs attention")
        if ph < 6.0:
            recs.append("Apply lime to raise soil pH")
        if ph > 7.5:
            recs.append("Apply sulfur/gypsum to lower soil pH")

        result = {
            "analysis": "NPK Soil Nutrient Analysis",
            "plot_id": plot_id,
            "nutrients": {
                "nitrogen_kg_per_ha": round(nitrogen, 2),
                "phosphorus_kg_per_ha": round(phosphorus, 2),
                "potassium_kg_per_ha": round(potassium, 2),
                "ph": round(ph, 2),
                "cec": round(cec, 2),
                "organic_carbon_pct": round(oc, 2),
            },
            "days_since_plantation": npk.get("days_since_plantation", 0),
            "recommendations": recs if recs else ["Nutrient levels are adequate"],
            "ideal_ranges": {
                "nitrogen": ">150 kg/ha",
                "phosphorus": ">30 kg/ha",
                "potassium": ">80 kg/ha",
                "ph": "6.0–7.5"
            }
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching nutrient data: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 6 — Weather Data
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_weather(lat: float, lon: float) -> str:
    """
    Fetches current real-time weather for field location using OpenWeatherMap.
    Use when user asks about: weather, temperature, rain, wind, humidity,
    should I spray today, is it going to rain, climate, forecast.
    """
    try:
        if not settings.openweather_api_key:
            return "Weather API key not configured."

        url = "https://api.openweathermap.org/data/2.5/weather"
        resp = await _client.get(url, params={
            "lat": lat, "lon": lon,
            "appid": settings.openweather_api_key,
            "units": "metric"
        })
        d = resp.json()

        main = d.get("main", {})
        wind = d.get("wind", {})
        weather_list = d.get("weather", [{}])
        rain = d.get("rain", {})

        wind_speed = wind.get("speed", 0)
        spray_safe = wind_speed < 8  # safe to spray below 8 m/s

        result = {
            "analysis": "Current Weather",
            "location": d.get("name", f"{lat},{lon}"),
            "temperature_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity_pct": main.get("humidity"),
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind.get("deg", 0),
            "description": weather_list[0].get("description", ""),
            "cloud_cover_pct": d.get("clouds", {}).get("all", 0),
            "rainfall_last_1h_mm": rain.get("1h", 0),
            "farming_advice": {
                "spray_safe": spray_safe,
                "spray_advice": "Safe to spray fertiliser/pesticide" if spray_safe else "Too windy — avoid spraying",
                "irrigation_needed": rain.get("1h", 0) < 2 and main.get("humidity", 100) < 70,
            }
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 7 — Registered Plots Info
# ═══════════════════════════════════════════════════════════════════════════
@tool
async def get_plot_info(plot_id: str, access_token: str = "") -> str:
    """
    Fetches basic info about a registered farm plot from the CropEye backend.
    Use when user asks about: my field, plot details, field area, crop name,
    plantation date, field location.
    """
    try:
        headers = {}
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        url = f"{settings.cropeye_backend_url}/api/farmers/plots/list/"
        resp = await _client.get(url, headers=headers)

        if resp.status_code != 200:
            return f"Could not fetch plot info (status {resp.status_code})"

        plots = resp.json()
        if isinstance(plots, list):
            plot = next((p for p in plots if
                         str(p.get("id")) == str(plot_id) or
                         p.get("name") == plot_id or
                         str(p.get("field_id")) == str(plot_id)), None)
            if plot:
                crops = plot.get("crops", [])
                crop_name = crops[0].get("crop_type_name", "Unknown") if crops else "Unknown"
                return json.dumps({
                    "plot_id": plot.get("id"),
                    "name": plot.get("name"),
                    "crop": crop_name,
                    "location": plot.get("location"),
                })
        return f"Plot {plot_id} not found in registered plots."
    except Exception as e:
        return f"Error fetching plot info: {str(e)}"


# All tools list — passed to LangGraph agent
ALL_TOOLS = [
    get_water_uptake_map,
    get_growth_map,
    get_pest_map,
    get_soil_moisture,
    get_soil_moisture_map,
    get_nutrient_analysis,
    get_weather,
    get_plot_info,
]
