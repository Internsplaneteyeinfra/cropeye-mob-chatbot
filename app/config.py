from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # gemini_api_key: str = ""
    # gemini_model: str = "gemini-2.5-flash"

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    plot_layer_base_url: str = "https://cropeyeappapis.up.railway.app"
    soil_param_api_url: str = "http://192.168.42.132:1000"
    cropeye_backend_url: str = "https://cropeye-mobilebackend.onrender.com"
    openweather_api_key: str = ""


settings = Settings()
