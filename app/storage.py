import shutil
from pathlib import Path


class FileStorage:
    """Stores file bytes on disk: <root>/<doc_id>/v<N><ext>.

    Only the bytes live here; everything *about* a document (name, versions,
    sizes) is in PostgreSQL. File names are chosen by us, never by the client,
    so a malicious filename can't escape the storage folder.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, doc_id: str, version: int, ext: str, data: bytes) -> str:
        """Write the bytes and return the path relative to root (stored in the DB)."""
        rel = Path(doc_id) / f"v{version}{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(rel)

    def read(self, rel_path: str) -> bytes:
        return (self.root / rel_path).read_bytes()

    def delete_document(self, doc_id: str) -> None:
        shutil.rmtree(self.root / doc_id, ignore_errors=True)
