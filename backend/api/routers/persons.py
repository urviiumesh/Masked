import os

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from api.config import FACE_DB_ROOT, TEMP_FACE_DB_ROOT
from api.services import face_service

router = APIRouter(prefix="/api/persons", tags=["persons"])


@router.get("")
def list_persons(search: str = ""):
    return face_service.list_persons(search)


@router.get("/{name}")
def get_person(name: str):
    person = face_service.get_person(name)
    if not person:
        raise HTTPException(404, "Person not found")
    return person


@router.post("")
async def create_person(name: str = Form(...), image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename or ".jpg")[1] or ".jpg"
    content = await image.read()
    path = face_service.save_upload(content, ext)
    try:
        result = face_service.create_person(name.strip(), path)
        return result
    except Exception as e:
        raise HTTPException(400, str(e))


@router.delete("/{name}")
def delete_person(name: str):
    if not face_service.delete_person(name):
        raise HTTPException(404, "Person not found")
    return {"deleted": name}


@router.put("/{name}")
def rename_person(name: str, new_name: str = Query(...)):
    if not face_service.rename_person(name, new_name.strip()):
        raise HTTPException(400, "Rename failed")
    return {"name": new_name}


@router.get("/{name}/thumbnail")
def thumbnail(name: str):
    for root in (FACE_DB_ROOT, TEMP_FACE_DB_ROOT):
        folder = os.path.join(root, name)
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_occluded" not in f:
                    return FileResponse(os.path.join(folder, f))
    raise HTTPException(404, "Thumbnail not found")


@router.get("/{name}/images/{filename}")
def person_image(name: str, filename: str):
    for root in (FACE_DB_ROOT, TEMP_FACE_DB_ROOT):
        path = os.path.join(root, name, filename)
        if os.path.isfile(path):
            return FileResponse(path)
    raise HTTPException(404, "Image not found")
