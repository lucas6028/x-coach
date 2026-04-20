import argparse
import json
import shutil
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "data" / "paper_metadata.json"
DEFAULT_RAG_DOCS = PROJECT_ROOT / "data" / "rag" / "docs"

def main():
    parser = argparse.ArgumentParser(description="Add a new document to the RAG vector DB with metadata.")
    parser.add_argument("filepath", type=Path, help="Path to the document to add (.txt, .md, .pdf, etc.)")
    parser.add_argument("--type", choices=["academic_paper", "health_blog", "forum", "training_guide", "textbook_excerpt", "encyclopedia", "other"], default="other", help="Source type")
    parser.add_argument("--ref", type=str, help="Full reference citation string. If not provided, filename is used.", default="")
    parser.add_argument("--title", type=str, help="Title of the paper/article", default="")
    parser.add_argument("--author", type=str, action="append", default=[], help="Author(s). Can specify multiple times.")
    parser.add_argument("--year", type=int, help="Publication year")
    
    args = parser.parse_args()

    if not args.filepath.exists():
        print(f"Error: File {args.filepath} does not exist.")
        sys.exit(1)

    # 1. Copy the file into the docs directory
    dest_path = DEFAULT_RAG_DOCS / args.filepath.name
    if dest_path.exists():
        print(f"Warning: {args.filepath.name} already exists in {DEFAULT_RAG_DOCS}. Overwriting...")
    
    shutil.copy2(args.filepath, dest_path)
    print(f"✅ Copied {args.filepath.name} to {DEFAULT_RAG_DOCS}")

    # 2. Update metadata JSON
    metadata = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Build the new entry
    new_entry = {
        "source_type": args.type,
        "reference": args.ref if args.ref else f"Source: {args.filepath.name}"
    }
    if args.title: new_entry["paper_title"] = args.title
    if args.author: new_entry["authors"] = args.author
    if args.year: new_entry["year"] = args.year

    metadata[args.filepath.name] = new_entry

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Added metadata for {args.filepath.name} to {METADATA_FILE.relative_to(PROJECT_ROOT)}")
    print(json.dumps(new_entry, indent=2))
    
    print("\nNext steps: Rebuild the Vector DB using:")
    print("python scripts/build_rag_vector_db.py build")

if __name__ == "__main__":
    main()
