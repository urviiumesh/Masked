import asyncio
import json
import time

from fastapi import APIRouter, Body, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from api.services.stream_service import stream_manager
from api.services.log_service import get_logs, export_xlsx

router = APIRouter(tags=["stream"])


class ConnectRequest(BaseModel):
    source: str | None = None
    preset_id: str | None = None
    location: str = "Live Feed"
    targets: list[str] | None = None


@router.get("/api/stream/presets")
def list_presets():
    from api.services.camera_presets import CAMERA_PRESETS, preset_public_view
    return [preset_public_view(p) for p in CAMERA_PRESETS]


@router.post("/api/stream/connect")
def connect_stream(req: ConnectRequest):
    try:
        if not req.source and not req.preset_id:
            from fastapi import HTTPException
            raise HTTPException(400, "Provide source or preset_id")
        return stream_manager.connect(
            req.source or "",
            req.location,
            req.targets,
            req.preset_id,
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(400, str(e))


@router.post("/api/stream/disconnect")
def disconnect_stream():
    stream_manager.disconnect()
    return {"connected": False}


@router.get("/api/stream/status")
def stream_status():
    return stream_manager.status()


@router.put("/api/stream/targets")
def set_targets(targets: list[str] = Body(...)):
    stream_manager.set_targets(targets)
    return stream_manager.status()


def _mjpeg_generator():
    last_seq = -1
    while stream_manager.status()["connected"]:
        jpeg, seq = stream_manager.get_jpeg()
        if jpeg is not None and seq != last_seq:
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        time.sleep(0.033)


@router.get("/api/stream/snapshot")
def stream_snapshot():
    jpeg, seq = stream_manager.get_jpeg()
    if jpeg is None:
        raise HTTPException(404, "No frame available")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Frame-Seq": str(seq),
        },
    )


@router.get("/api/stream/mjpeg")
def mjpeg_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/api/stream/ws")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    q = stream_manager.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=1.0)
                await websocket.send_text(json.dumps(event))
            except asyncio.TimeoutError:
                if not stream_manager.status()["connected"]:
                    break
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        stream_manager.unsubscribe(q)


@router.websocket("/api/stream/ws/frames")
async def stream_frames_ws(websocket: WebSocket):
    await websocket.accept()
    q = stream_manager.subscribe_frames()
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=1.0)
                jpeg = event.get("jpeg")
                seq = event.get("seq", 0)
                if jpeg:
                    await websocket.send_bytes(seq.to_bytes(4, "big") + jpeg)
            except asyncio.TimeoutError:
                if not stream_manager.status()["connected"]:
                    break
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        stream_manager.unsubscribe_frames(q)


@router.get("/api/logs")
def logs(limit: int = 50, source: str | None = None):
    return get_logs(limit, source)


@router.get("/api/logs/export")
def export_logs(source: str | None = None):
    path = export_xlsx(source)
    return FileResponse(path, filename="detection_logs.xlsx")


@router.get("/api/logs/snapshots/{filename}")
def snapshot(filename: str):
    from api.config import SNAPSHOTS_DIR
    import os
    path = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.isfile(path):
        from fastapi import HTTPException
        raise HTTPException(404)
    return FileResponse(path)
