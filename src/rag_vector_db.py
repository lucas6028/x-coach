from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "rag" / "vector_db"
DEFAULT_RAG_DOCS_DIR = PROJECT_ROOT / "data" / "rag" / "docs"
DEFAULT_KG_DOCS_DIR = PROJECT_ROOT / "data" / "kg" / "docs"
TEXT_SUFFIXES = {".txt", ".md", ".html", ".htm", ".json"}
ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp950", "big5", "gb18030")


def get_default_sources() -> list[Path]:
    sources: list[Path] = []
    if DEFAULT_RAG_DOCS_DIR.exists():
        sources.append(DEFAULT_RAG_DOCS_DIR)
    if DEFAULT_KG_DOCS_DIR.exists():
        sources.append(DEFAULT_KG_DOCS_DIR)
    return sources


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


class HashEmbeddingBackend:
    """A tiny local embedding backend so the repo can build a vector DB offline."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenize(text):
                index = self._hash_to_index(token)
                matrix[row, index] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower()
        words = re.findall(r"\w+", normalized, flags=re.UNICODE)
        bigrams = [
            normalized[idx : idx + 2]
            for idx in range(max(len(normalized) - 1, 0))
            if not normalized[idx : idx + 2].isspace()
        ]
        return words + bigrams

    def _hash_to_index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dimensions


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ENCODING_CANDIDATES:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def graph_to_documents(path: Path) -> list[tuple[str, dict[str, Any]]]:
    graph = nx.read_graphml(path)
    docs: list[tuple[str, dict[str, Any]]] = []

    for node_id, attrs in graph.nodes(data=True):
        label = attrs.get("label", "Entity")
        text = f"Entity: {node_id}\nType: {label}"
        docs.append(
            (
                text,
                {
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "kind": "kg_node",
                    "node_id": str(node_id),
                    "node_label": str(label),
                },
            )
        )

    for source, target, attrs in graph.edges(data=True):
        relation = attrs.get("type", "RELATED_TO")
        source_label = graph.nodes[source].get("label", "Entity")
        target_label = graph.nodes[target].get("label", "Entity")
        text = (
            f"Knowledge triple: {source} ({source_label}) {relation} "
            f"{target} ({target_label})"
        )
        docs.append(
            (
                text,
                {
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "kind": "kg_edge",
                    "source_id": str(source),
                    "target_id": str(target),
                    "relation": str(relation),
                },
            )
        )

    return docs


def expand_sources(sources: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        if not source.exists():
            continue
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES.union({".pdf"}):
                    files.append(path)
        else:
            files.append(source)
    return files


def load_source_documents(path: Path) -> list[tuple[str, dict[str, Any]]]:
    suffix = path.suffix.lower()
    relative = str(path.relative_to(PROJECT_ROOT))

    if suffix == ".pdf":
        return [(load_pdf_text(path), {"source": relative, "kind": "pdf"})]
    if suffix == ".graphml":
        return graph_to_documents(path)

    text = read_text_file(path)
    return [(text, {"source": relative, "kind": "text"})]


def chunk_documents(
    documents: list[tuple[str, dict[str, Any]]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks: list[ChunkRecord] = []

    for text, metadata in documents:
        pieces = splitter.split_text(text)
        for index, piece in enumerate(pieces):
            cleaned = piece.strip()
            if not cleaned:
                continue
            chunk_id = hashlib.md5(
                f"{metadata['source']}::{metadata.get('kind', 'text')}::{index}::{cleaned}".encode(
                    "utf-8"
                )
            ).hexdigest()
            enriched_metadata = dict(metadata)
            enriched_metadata["chunk_index"] = index
            enriched_metadata["char_length"] = len(cleaned)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=cleaned,
                    metadata=enriched_metadata,
                )
            )
    return chunks


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def get_file_hash(path: Path) -> str:
    """Get a hash of file content for change detection."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_existing_db_state(db_dir: Path) -> dict[str, str]:
    """Load the file state tracking from manifest."""
    manifest_path = db_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("file_hashes", {})
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def identify_new_or_modified_files(
    files: list[Path],
    file_states: dict[str, str],
) -> list[Path]:
    """Identify files that are new or have been modified."""
    new_files = []
    for file_path in files:
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        current_hash = get_file_hash(file_path)

        if relative_path not in file_states or file_states[relative_path] != current_hash:
            new_files.append(file_path)

    return new_files


def build_vector_db(
    *,
    source_paths: list[Path] | None = None,
    db_dir: Path = DEFAULT_DB_DIR,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    incremental: bool = True,
) -> dict[str, Any]:
    source_paths = source_paths or get_default_sources()
    files = expand_sources(source_paths)

    # Load existing state if incremental mode is enabled
    existing_chunks: list[ChunkRecord] = []
    existing_vectors: np.ndarray | None = None
    existing_file_hashes: dict[str, str] = {}

    if incremental and (db_dir / "chunks.json").exists():
        try:
            existing_chunks, existing_vectors, manifest = load_vector_db(db_dir)
            existing_file_hashes = manifest.get("file_hashes", {})
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass

    # Identify files that are new or modified
    files_to_process = identify_new_or_modified_files(files, existing_file_hashes)

    if not files_to_process:
        # No new files, return existing stats
        return {
            "db_dir": str(db_dir),
            "source_count": len(files),
            "chunk_count": len(existing_chunks),
            "embedding_backend": "hash-384",
            "status": "up_to_date",
        }

    # Process only new/modified files
    new_documents: list[tuple[str, dict[str, Any]]] = []
    for path in files_to_process:
        new_documents.extend(load_source_documents(path))

    new_chunks = chunk_documents(
        new_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Embed new chunks
    backend = HashEmbeddingBackend()
    new_vectors = backend.embed_documents([chunk.text for chunk in new_chunks])

    # Merge with existing chunks and vectors
    all_chunks = existing_chunks + new_chunks
    if existing_vectors is not None:
        all_vectors = np.vstack([existing_vectors, new_vectors])
    else:
        all_vectors = new_vectors

    # Update file hashes
    updated_file_hashes = dict(existing_file_hashes)
    for path in files_to_process:
        relative_path = str(path.relative_to(PROJECT_ROOT))
        updated_file_hashes[relative_path] = get_file_hash(path)

    # Write the merged database
    db_dir.mkdir(parents=True, exist_ok=True)
    np.save(db_dir / "embeddings.npy", all_vectors)
    write_json(db_dir / "chunks.json", [chunk.to_json() for chunk in all_chunks])
    write_json(
        db_dir / "manifest.json",
        {
            "embedding_backend": backend.name,
            "dimensions": backend.dimensions,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "source_count": len(files),
            "chunk_count": len(all_chunks),
            "sources": [str(path.relative_to(PROJECT_ROOT)) for path in files],
            "file_hashes": updated_file_hashes,
        },
    )

    return {
        "db_dir": str(db_dir),
        "source_count": len(files),
        "chunk_count": len(all_chunks),
        "new_chunks_added": len(new_chunks),
        "embedding_backend": backend.name,
        "status": "updated",
    }


def load_vector_db(db_dir: Path = DEFAULT_DB_DIR) -> tuple[list[ChunkRecord], np.ndarray, dict[str, Any]]:
    chunks_payload = json.loads((db_dir / "chunks.json").read_text(encoding="utf-8"))
    manifest = json.loads((db_dir / "manifest.json").read_text(encoding="utf-8"))
    vectors = np.load(db_dir / "embeddings.npy")
    chunks = [ChunkRecord(**item) for item in chunks_payload]
    return chunks, vectors, manifest


def query_vector_db(
    query: str,
    *,
    db_dir: Path = DEFAULT_DB_DIR,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    chunks, vectors, manifest = load_vector_db(db_dir)
    backend = HashEmbeddingBackend(dimensions=int(manifest["dimensions"]))
    query_vector = backend.embed_query(query)

    scores = vectors @ query_vector
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        score = float(scores[index])
        if math.isclose(score, 0.0, abs_tol=1e-8):
            continue
        results.append(
            {
                "rank": rank,
                "score": round(score, 4),
                "chunk_id": chunks[index].chunk_id,
                "text": chunks[index].text,
                "metadata": chunks[index].metadata,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or query the local RAG vector DB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the local vector DB.")
    build_parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    build_parser.add_argument("--chunk-size", type=int, default=900)
    build_parser.add_argument("--chunk-overlap", type=int, default=180)
    build_parser.add_argument(
        "--source",
        type=Path,
        action="append",
        help="Override default source paths. Can be passed multiple times.",
    )
    build_parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Force a full rebuild instead of incremental update.",
    )

    query_parser = subparsers.add_parser("query", help="Query the local vector DB.")
    query_parser.add_argument("query", type=str)
    query_parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    query_parser.add_argument("--top-k", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build":
        result = build_vector_db(
            source_paths=args.source,
            db_dir=args.db_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            incremental=not args.full_rebuild,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "query":
        results = query_vector_db(args.query, db_dir=args.db_dir, top_k=args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
