"""
CropEye Chatbot — FastAPI entry point
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8001
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(
    title="CropEye AI Chatbot",
    description="Multilingual farming assistant powered by Gemini + LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "cropeye-chatbot"}
