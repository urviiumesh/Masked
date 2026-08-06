import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routers import persons, stream, video

app = FastAPI(title="DHRISHTI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(persons.router)
app.include_router(stream.router)
app.include_router(video.router)


@app.get("/api/health")
def health():
    from api.services.runtime_device import runtime_info
    info = runtime_info()
    return {
        "status": "ok",
        "service": "dhrishti",
        "gpu_enabled": info["gpu_enabled"],
        "ort_provider": info["active_provider"],
        "available_providers": info["available_providers"],
        "hint": info["hint"],
    }


FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend", "dhrishti", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
