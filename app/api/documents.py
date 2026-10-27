from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.chunking import chunk_pages
from app.config import Settings, get_settings
from app.db import get_db
from app.embeddings import EmbedderDep
from app.extraction import ExtractionError
from app.models import Document, DocumentVersion, _now
from app.schemas import ChunkedDocument, DocumentInfo, ExtractedText, IndexResult, IndexStatus
from app.services import find_version, list_documents, read_pages, to_info
from app.storage import FileStorage
from app.vectorstore import VectorStoreDep

router = APIRouter(prefix="/documents", tags=["documents"])


def get_storage(settings: Annotated[Settings, Depends(get_settings)]) -> FileStorage:
    return FileStorage(settings.data_dir / "uploads")


StorageDep = Annotated[FileStorage, Depends(get_storage)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]


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
def list_all_documents(db: DbDep) -> list[DocumentInfo]:
    return list_documents(db)


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str, db: DbDep) -> DocumentInfo:
    return to_info(get_or_404(db, doc_id))


def pick_version(doc: Document, version: int | None) -> DocumentVersion:
    """The requested version, or the latest when none is given."""
    chosen = find_version(doc, version)
    if chosen is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Version {version} not found")
    return chosen


def extract_version(storage: FileStorage, chosen: DocumentVersion):
    try:
        return read_pages(storage, chosen)
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


@router.post("/{doc_id}/index", response_model=IndexResult)
def index_document(
    doc_id: str,
    db: DbDep,
    storage: StorageDep,
    settings: SettingsDep,
    embedder: EmbedderDep,
    vectors: VectorStoreDep,
    version: int | None = None,
) -> IndexResult:
    """Extract, chunk, embed and store a document's vectors (latest version by default).

    Re-indexing replaces whatever was indexed for this document before.
    """
    doc = get_or_404(db, doc_id)
    chosen = pick_version(doc, version)
    pages = extract_version(storage, chosen)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    vecs = embedder.embed([c.text for c in chunks]) if chunks else []
    count = vectors.replace_document(doc.id, chosen.version, doc.filename, chunks, vecs)
    return IndexResult(
        document_id=doc.id,
        version=chosen.version,
        chunks_indexed=count,
        collection=vectors.collection,
    )


@router.get("/{doc_id}/index", response_model=IndexStatus)
def index_status(doc_id: str, db: DbDep, vectors: VectorStoreDep) -> IndexStatus:
    get_or_404(db, doc_id)
    count = vectors.count(doc_id)
    return IndexStatus(
        document_id=doc_id,
        indexed=count > 0,
        chunk_count=count,
        version=vectors.indexed_version(doc_id) if count else None,
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: str, db: DbDep, storage: StorageDep, vectors: VectorStoreDep
) -> None:
    doc = get_or_404(db, doc_id)
    db.delete(doc)
    db.commit()
    storage.delete_document(doc_id)
    vectors.delete_document(doc_id)
