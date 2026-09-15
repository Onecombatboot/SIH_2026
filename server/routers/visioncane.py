from fastapi import APIRouter, WebSocket
from server.ws import stream

router = APIRouter()

_camera    = None
_depth     = None
_processor = None


def _get_deps():
    global _camera, _depth, _processor
    # Heavy imports deferred to first connection — keeps server startup instant
    from core.camera import Camera
    from core.depthmap import DepthEstimator
    from apps.visioncane.processor import VisionCaneProcessor
    from apps.visioncane.config import DEFAULT_MODEL_ID, DEFAULT_CAM_INDEX, DEFAULT_YOLO_WEIGHTS

    if _camera is None:
        _camera = Camera(index=DEFAULT_CAM_INDEX)
    if _depth is None:
        _depth = DepthEstimator(model_id=DEFAULT_MODEL_ID)
    if _processor is None:
        _processor = VisionCaneProcessor(yolo_weights=DEFAULT_YOLO_WEIGHTS)
    return _camera, _depth, _processor


@router.get("/config")
def config():
    from apps.visioncane.config import DEFAULT_MODEL_ID, DEFAULT_YOLO_WEIGHTS, DEFAULT_CAM_INDEX
    return {"app": "visioncane", "model": DEFAULT_MODEL_ID, "yolo": DEFAULT_YOLO_WEIGHTS, "cam": DEFAULT_CAM_INDEX}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        camera, depth, processor = _get_deps()
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1011)
        return
    await stream(websocket, camera, depth, processor)
