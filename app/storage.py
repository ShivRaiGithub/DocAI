import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.schemas import DocumentInfo

META_FILE = "meta.json"


class FileStorage:
    """Stores each document in its own folder: <root>/<id>/{content<ext>, meta.json}.

    The file on disk is named by us, never by the client, so a malicious
    filename like "../../etc/passwd" can't escape the storage folder.
    Metadata lives in meta.json until Stage 3 moves it to PostgreSQL.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self, filename: str, content_type: str | None, data: bytes
    ) -> DocumentInfo:
        doc_id = uuid.uuid4().hex
        doc_dir = self.root / doc_id
        doc_dir.mkdir()
        (doc_dir / f"content{Path(filename).suffix.lower()}").write_bytes(data)
        info = DocumentInfo(
            id=doc_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            uploaded_at=datetime.now(UTC),
        )
        (doc_dir / META_FILE).write_text(info.model_dump_json())
        return info

    def get(self, doc_id: str) -> DocumentInfo | None:
        meta = self._dir(doc_id) / META_FILE
        if not meta.is_file():
            return None
        return DocumentInfo.model_validate(json.loads(meta.read_text()))

    def list(self) -> list[DocumentInfo]:
        docs = [self.get(d.name) for d in self.root.iterdir() if d.is_dir()]
        return sorted(
            (d for d in docs if d), key=lambda d: d.uploaded_at, reverse=True
        )

    def delete(self, doc_id: str) -> bool:
        doc_dir = self._dir(doc_id)
        if not (doc_dir / META_FILE).is_file():
            return False
        shutil.rmtree(doc_dir)
        return True

    def _dir(self, doc_id: str) -> Path:
        # doc_id comes from the URL, so reject anything that isn't a bare name
        if Path(doc_id).name != doc_id:
            return self.root / "_invalid"
        return self.root / doc_id
