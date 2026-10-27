from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.models import Document, DocumentVersion, _now
from app.chunking import chunk_pages
from app.extraction import ExtractionError, extract_pages
from app.schemas import ChunkedDocument, DocumentInfo, ExtractedText
from app.storage import FileStorage

router = APIRouter(prefix="/documents", tags=["documents"])


def get_storage(settings: Annotated[Settings, Depends(get_settings)]) -> FileStorage:
    return FileStorage(settings.data_dir / "uploads")


StorageDep = Annotated[FileStorage, Depends(get_storage)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]


def to_info(doc: Document) -> DocumentInfo:
    latest = doc.latest
    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        version=latest.version,
        size_bytes=latest.size_bytes,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


async def read_valid_upload(file: UploadFile, settings: Settings) -> tuple[str, str, bytes]:
    """Validate an upload; returns (clean filename, extension, bytes)."""
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
    return filename, ext, data


def get_or_404(db: Session, doc_id: str) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


@router.post("", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile, db: DbDep, storage: StorageDep, settings: SettingsDep
) -> DocumentInfo:
    filename, ext, data = await read_valid_upload(file, settings)
    doc = Document(filename=filename, content_type=file.content_type)
    db.add(doc)
    db.flush()  # assigns doc.id without committing yet
    path = storage.save(doc.id, 1, ext, data)
    doc.versions.append(DocumentVersion(version=1, size_bytes=len(data), storage_path=path))
    db.commit()
    return to_info(doc)


@router.post("/{doc_id}/versions", response_model=DocumentInfo, status_code=status.HTTP_201_CREATED)
async def upload_new_version(
    doc_id: str, file: UploadFile, db: DbDep, storage: StorageDep, settings: SettingsDep
) -> DocumentInfo:
    doc = get_or_404(db, doc_id)
    filename, ext, data = await read_valid_upload(file, settings)
    version = doc.latest.version + 1
    path = storage.save(doc.id, version, ext, data)
    doc.versions.append(DocumentVersion(version=version, size_bytes=len(data), storage_path=path))
    doc.filename = filename
    doc.content_type = file.content_type
    doc.updated_at = _now()
    db.commit()
    return to_info(doc)


@router.get("", response_model=list[DocumentInfo])
def list_documents(db: DbDep) -> list[DocumentInfo]:
    docs = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [to_info(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, db: DbDep) -> DocumentInfo:
    return to_info(get_or_404(db, doc_id))


def pick_version(doc: Document, version: int | None) -> DocumentVersion:
    """The requested version, or the latest when none is given."""
    if version is None:
        return doc.latest
    chosen = next((v for v in doc.versions if v.version == version), None)
    if chosen is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Version {version} not found")
    return chosen


def extract_version(storage: FileStorage, chosen: DocumentVersion):
    try:
        data = storage.read(chosen.storage_path)
        return extract_pages(data, Path(chosen.storage_path).suffix)
    except ExtractionError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e


@router.get("/{doc_id}/text", response_model=ExtractedText)
def get_document_text(
    doc_id: str, db: DbDep, storage: StorageDep, version: int | None = None
) -> ExtractedText:
    """Extract and return the text of a document (latest version unless ?version=N)."""
    doc = get_or_404(db, doc_id)
    chosen = pick_version(doc, version)
    pages = extract_version(storage, chosen)
    return ExtractedText(
        document_id=doc.id,
        version=chosen.version,
        filename=doc.filename,
        page_count=len(pages),
        char_count=sum(len(p.text) for p in pages),
        pages=pages,
    )


@router.get("/{doc_id}/chunks", response_model=ChunkedDocument)
def get_document_chunks(
    doc_id: str,
    db: DbDep,
    storage: StorageDep,
    settings: SettingsDep,
    version: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> ChunkedDocument:
    """Split a document into overlapping chunks. Sizes default to the app settings."""
    doc = get_or_404(db, doc_id)
    chosen = pick_version(doc, version)
    size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    pages = extract_version(storage, chosen)
    try:
        chunks = chunk_pages(pages, size, overlap)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    return ChunkedDocument(
        document_id=doc.id,
        version=chosen.version,
        chunk_size=size,
        chunk_overlap=overlap,
        chunk_count=len(chunks),
        chunks=chunks,
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, db: DbDep, storage: StorageDep) -> None:
    doc = get_or_404(db, doc_id)
    db.delete(doc)
    db.commit()
    storage.delete_document(doc_id)
