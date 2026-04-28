from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    plot_id: str = Field(..., description="Plot/field identifier (plot_name)")
    user_message: str = Field(..., description="User's message in any language")
    session_id: str = Field(..., description="Unique session ID per user (use phone number or UUID)")
    farmer_name: Optional[str] = Field(None, description="Farmer's name for personalisation")
    plantation_date: Optional[str] = Field(None, description="Plantation date YYYY-MM-DD")
    crop_type: Optional[str] = Field(None, description="Crop type e.g. Wheat, Grape")
    lat: Optional[float] = Field(None, description="Field latitude")
    lon: Optional[float] = Field(None, description="Field longitude")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Bot reply in same language as user")
    session_id: str
    plot_id: str
    data_sources_used: list[str] = Field(
        default_factory=list,
        description="Which APIs were called to answer this question"
    )


class ClearHistoryRequest(BaseModel):
    session_id: str
    plot_id: str
