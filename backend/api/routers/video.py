import os

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.services import face_service, video_service

router = APIRouter(prefix="/api/video", tags=["video"])


class RescanRequest(BaseModel):
    targets: list[str]


@router.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    targets: str = Form(""),
):
    ext = os.path.splitext(video.filename or ".mp4")[1] or ".mp4"
    content = await video.read()
    path = face_service.save_upload(content, ext)
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    if not target_list:
        raise HTTPException(400, "Select at least one target before processing")
    job_id = video_service.create_job(path, target_list, video.filename or "video.mp4")
    return {"job_id": job_id, "filename": video.filename or "video.mp4", "targets": target_list}


@router.get("/jobs")
def list_jobs():
    return video_service.list_jobs()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = video_service.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/output")
def download_output(job_id: str):
    job = video_service.get_job_raw(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Output not ready")
    path = job.get("output_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "Output file not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="processed_{job_id}.mp4"',
            "Cache-Control": "no-cache",
        },
    )


@router.post("/jobs/{job_id}/rescan")
def rescan_job(job_id: str, req: RescanRequest):
    target_list = [t.strip() for t in req.targets if t and str(t).strip()]
    if not target_list:
        raise HTTPException(400, "Select at least one target before rescanning")
    try:
        return video_service.rescan_job(job_id, target_list)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    try:
        ok = video_service.delete_job(job_id)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if not ok:
        raise HTTPException(404, "Job not found")
    return {"deleted": job_id}
