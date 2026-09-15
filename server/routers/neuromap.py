from fastapi import APIRouter, WebSocket
from server.ws import stream_game

router = APIRouter()

_camera    = None
_processor = None


def _get_deps():
    global _camera, _processor
    from core.camera import Camera
    from apps.neuromap.processor import NeuroMapProcessor
    from apps.neuromap.config import DEFAULT_CAM_INDEX

    if _camera is None:
        _camera = Camera(index=DEFAULT_CAM_INDEX)
    if _processor is None:
        _processor = NeuroMapProcessor()
    return _camera, _processor


@router.get("/config")
def config():
    from apps.neuromap.config import DEFAULT_CAM_INDEX
    return {"app": "neuromap", "cam": DEFAULT_CAM_INDEX}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        camera, processor = _get_deps()
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1011)
        return
    await stream_game(websocket, camera, processor)
