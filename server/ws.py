import asyncio
import base64

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from core.utils import depth_to_proximity, to_u8_normalized, resize_keep_aspect


async def stream(
    websocket: WebSocket,
    frame_source,       # Camera or VideoFile — both expose .read()
    depth_estimator,
    processor,
) -> None:
    """
    General-purpose streaming loop for depth-based apps (VisionCane, GhostNav).

    Reads processor.colormap and processor.smooth_sigma if present, so each
    app can declare its own terrain colormap and depth smoothing without
    touching this function.
    """
    colormap = getattr(processor, "colormap", cv2.COLORMAP_MAGMA)
    smooth_sigma = getattr(processor, "smooth_sigma", 0.0)
    invert = getattr(processor, "invert", True)

    try:
        while True:
            frame_bgr = frame_source.read()
            if frame_bgr is None:
                await asyncio.sleep(0.01)
                continue

            proc_bgr, _, _ = resize_keep_aspect(frame_bgr, processor.proc_short)
            proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)

            depth = await asyncio.to_thread(depth_estimator.infer, proc_rgb)
            cue   = await asyncio.to_thread(processor.process, proc_bgr, depth)

            # Optional Gaussian smoothing (terrain visualization)
            disp_depth = depth
            if smooth_sigma > 0:
                ksize = int(smooth_sigma * 3) * 2 + 1
                disp_depth = cv2.GaussianBlur(depth, (ksize, ksize), smooth_sigma)

            # Encode camera frame
            _, frame_jpg = cv2.imencode(".jpg", proc_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(frame_jpg.tobytes()).decode()

            # Encode depth colormap
            prox_u8 = to_u8_normalized(depth_to_proximity(disp_depth, invert=invert))
            depth_color = cv2.applyColorMap(prox_u8, colormap)
            _, depth_jpg = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 75])
            depth_b64 = base64.b64encode(depth_jpg.tobytes()).decode()

            await websocket.send_json({
                "frame": frame_b64,
                "depth": depth_b64,
                "cue":   cue,
            })

    except WebSocketDisconnect:
        pass


async def stream_game(
    websocket: WebSocket,
    camera,
    processor,
) -> None:
    """
    Game-loop streaming for NeuroMap — no depth estimation.
    The processor handles face tracking, game state, and frame annotation
    internally, returning {"annotated_bgr", "cue", "game"} each frame.
    """
    try:
        while True:
            frame_bgr = camera.read()
            if frame_bgr is None:
                await asyncio.sleep(0.01)
                continue

            proc_bgr, _, _ = resize_keep_aspect(frame_bgr, processor.proc_short)

            result = await asyncio.to_thread(processor.process, proc_bgr)

            display = result.get("annotated_bgr", proc_bgr)
            _, jpg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(jpg.tobytes()).decode()

            await websocket.send_json({
                "frame": frame_b64,
                "depth": None,
                "cue":   result.get("cue", {"present": False}),
                "game":  result.get("game"),
            })

    except WebSocketDisconnect:
        pass
