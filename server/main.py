import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.routers import visioncane, neuromap


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded lazily on first WebSocket connection per app.
    # Move init here if startup pre-loading is preferred.
    yield


app = FastAPI(title="HEIMDALL", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(visioncane.router, prefix="/visioncane", tags=["visioncane"])
app.include_router(neuromap.router, prefix="/neuromap", tags=["neuromap"])

# Serve built React frontend in production (run `npm run build` first)
_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
