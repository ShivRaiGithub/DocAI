from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.schemas import DocumentInfo
from app.storage import FileStorage

router = APIRouter(prefix="/documents", tags=["documents"])


def get_storage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileStorage:
    return FileStorage(settings.data_dir / "uploads")


StorageDep = Annotated[FileStorage, Depends(get_storage)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.post("", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, storage: StorageDep, settings: SettingsDep
) -> DocumentInfo:
    filename = Path(file.filename or "").name
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File exceeds {settings.max_upload_mb} MB limit",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    return storage.save(filename, file.content_type, data)


@router.get("", response_model=list[DocumentInfo])
def list_documents(storage: StorageDep) -> list[DocumentInfo]:
    return storage.list()


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, storage: StorageDep) -> DocumentInfo:
    info = storage.get(doc_id)
    if info is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return info


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, storage: StorageDep) -> None:
    if not storage.delete(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
