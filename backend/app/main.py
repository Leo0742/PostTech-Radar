from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import PROJECT_ROOT

app = FastAPI(
    title="ПочтаТех Радар API",
    version="1.0.0",
    description="Локальная категоризация, маршрутизация, поиск похожих обращений и аналитика SLA.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
assets = frontend_dist / "assets"
if assets.exists():
    app.mount("/assets", StaticFiles(directory=assets), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def frontend(full_path: str) -> FileResponse:
    index = frontend_dist / "index.html"
    if not index.exists():
        return FileResponse(PROJECT_ROOT / "frontend" / "public" / "not-built.html", status_code=503)
    candidate = (frontend_dist / full_path).resolve()
    if candidate.is_file() and frontend_dist.resolve() in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(index)
