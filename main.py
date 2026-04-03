# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.text import router as text_router
from routers.image import router as image_router
from routers.video import router as video_router

app = FastAPI(title="AI vs Human Content Detection API")

# CORS (Required for your Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # when deploying, change to your domain only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(text_router)   # no prefix
app.include_router(image_router)  # no prefix
app.include_router(video_router)  # no prefix