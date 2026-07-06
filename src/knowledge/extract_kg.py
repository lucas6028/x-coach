import os
import sys
import json
import argparse
import re
from pydantic import BaseModel, Field
from typing import List

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openrouter import ChatOpenRouter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import networkx as nx
except ImportError:
    import subprocess
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "langchain", "langchain-core", "langchain-google-genai", 
        "langchain-community", "pydantic", "networkx", "pypdf", "beautifulsoup4"
    ])
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openrouter import ChatOpenRouter
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import networkx as nx

from src.knowledge.kg_schema import (
    DEFAULT_GRAPH_FILE as SCHEMA_GRAPH_FILE,
    canonical_shared_names,
    resolve_node_id,
)

DEFAULT_GRAPH_FILE = str(SCHEMA_GRAPH_FILE)

CANONICAL_NODE_LABELS = {
    "action": "Action",
    "phase": "Phase",
    "fault": "Fault",
    "error": "Fault",
    "commonfault": "Fault",
    "evidence": "EvidenceSignal",
    "evidencesignal": "EvidenceSignal",
    "signal": "EvidenceSignal",
    "cause": "Cause",
    "anatomy": "Cause",
    "anatomycause": "Cause",
    "risk": "Risk",
    "correction": "Cue",
    "cue": "Cue",
    "coachingcue": "Cue",
    "qualitydimension": "QualityDimension",
}

CANONICAL_EDGE_TYPES = {
    "hasphase": "HAS_PHASE",
    "hasfault": "HAS_FAULT",
    "haspotentialerror": "HAS_FAULT",
    "occursinphase": "OCCURS_IN_PHASE",
    "indicatedby": "INDICATED_BY",
    "causedby": "CAUSED_BY",
    "causedbyweaknessin": "CAUSED_BY",
    "increasesriskof": "INCREASES_RISK_OF",
    "correctedby": "CORRECTED_BY",
    "affectsquality": "AFFECTS_QUALITY",
}

PHASE_ALIASES = {
    "setup": "Setup",
    "walkout": "Setup",
    "descent": "Descent",
    "eccentric": "Descent",
    "bottom": "Bottom",
    "hole": "Bottom",
    "ascent": "Ascent",
    "concentric": "Ascent",
    "lockout": "Lockout",
}

NODE_ALIASES = {
    "front squat": "Front Squat",
    "high bar squat": "High Bar Squat",
    "high-bar squat": "High Bar Squat",
    "low bar squat": "Low Bar Squat",
    "low-bar squat": "Low Bar Squat",
    "bodyweight squat": "Bodyweight Squat",
    "body-weight squat": "Bodyweight Squat",
    "knees cave in": "Knee Valgus",
    "knee cave in": "Knee Valgus",
    "knees collapse inward": "Knee Valgus",
    "knees collapse in": "Knee Valgus",
    "torso rounding or flexing forward": "Lumbar Flexion",
    "rounded back": "Lumbar Flexion",
    "incomplete squat": "Shallow Depth",
    "shallow squat": "Shallow Depth",
    "heels rise": "Heel Rise",
    "heel rise": "Heel Rise",
    "heels lifting": "Heel Rise",
    "looking down": "Head Down",
}


def canonicalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def compact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonicalize_whitespace(text).lower())


def title_case_phrase(text: str) -> str:
    words = canonicalize_whitespace(text).split(" ")
    return " ".join(word if word.isupper() else word.capitalize() for word in words if word)


def normalize_node_label(label: str) -> str:
    normalized = canonicalize_whitespace(label)
    return CANONICAL_NODE_LABELS.get(compact_key(normalized), title_case_phrase(normalized))


def normalize_edge_type(edge_type: str) -> str:
    normalized = canonicalize_whitespace(edge_type)
    return CANONICAL_EDGE_TYPES.get(compact_key(normalized), normalized.replace("-", "_").replace(" ", "_").upper())


def normalize_node_id(node_id: str, label: str) -> str:
    normalized = canonicalize_whitespace(node_id)
    alias_key = normalized.lower()
    if label == "Phase":
        return PHASE_ALIASES.get(alias_key, title_case_phrase(normalized))
    return NODE_ALIASES.get(alias_key, title_case_phrase(normalized))


class Node(BaseModel):
    id: str = Field(description="Canonical identifier for the node (e.g., 'Squat', 'Knee Valgus').")
    label: str = Field(
        description=(
            "The node type. Must be one of: Action, Phase, Fault, EvidenceSignal, "
            "Cause, Risk, Cue, QualityDimension."
        )
    )

class Edge(BaseModel):
    source: str = Field(description="The id of the source node")
    target: str = Field(description="The id of the target node")
    type: str = Field(
        description=(
            "The relationship type. Must be one of: HAS_PHASE, HAS_FAULT, OCCURS_IN_PHASE, "
            "INDICATED_BY, CAUSED_BY, INCREASES_RISK_OF, CORRECTED_BY, AFFECTS_QUALITY."
        )
    )

class KnowledgeGraph(BaseModel):
    """A knowledge graph consisting of nodes and edges representing AQA-oriented sports biomechanics."""
    nodes: List[Node] = Field(description="List of entities (nodes) in the knowledge graph")
    edges: List[Edge] = Field(description="List of relationships (edges) in the knowledge graph")

def normalize_kg(kg: KnowledgeGraph) -> KnowledgeGraph:
    canonical_nodes: dict[str, Node] = {}
    original_to_canonical: dict[str, str] = {}
    original_labels: dict[str, str] = {}

    for node in kg.nodes:
        normalized_label = normalize_node_label(node.label)
        normalized_id = normalize_node_id(node.id, normalized_label)
        canonical_nodes[normalized_id] = Node(id=normalized_id, label=normalized_label)
        original_to_canonical[node.id] = normalized_id
        original_labels[node.id] = normalized_label

    canonical_edges: list[Edge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for edge in kg.edges:
        source_label = original_labels.get(edge.source, "Action")
        target_label = original_labels.get(edge.target, "Fault")
        normalized_source = original_to_canonical.get(edge.source, normalize_node_id(edge.source, source_label))
        normalized_target = original_to_canonical.get(edge.target, normalize_node_id(edge.target, target_label))
        normalized_type = normalize_edge_type(edge.type)
        edge_key = (normalized_source, normalized_target, normalized_type)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        canonical_edges.append(
            Edge(
                source=normalized_source,
                target=normalized_target,
                type=normalized_type,
            )
        )

    return KnowledgeGraph(nodes=list(canonical_nodes.values()), edges=canonical_edges)


def update_property_graph(kg: KnowledgeGraph, graph_file=DEFAULT_GRAPH_FILE, movement: str = "Squat"):
    """Adds the extracted Knowledge Graph data into a persistent NetworkX GraphML file.

    Node ids are placed in the multi-movement schema (docs/kg-schema-generalization.md):
    scoped labels become `Movement:Name` (movement=<movement>); shared labels collapse
    to their canonical vocab name (movement="shared"). See kg_schema.resolve_node_id.
    """
    kg = normalize_kg(kg)
    if os.path.exists(graph_file):
        G = nx.read_graphml(graph_file)
        # GraphML may deserialize to DiGraph if no parallel edges exist.
        # Convert to MultiDiGraph so relationship deduplication by `type` is consistent.
        if not G.is_multigraph():
            G = nx.MultiDiGraph(G)
        print(f"Loaded existing graph from {graph_file} with {G.number_of_nodes()} nodes.")
    else:
        G = nx.MultiDiGraph()
        print(f"Created new Labeled Property Graph.")

    # Map each extracted node id -> schema graph id, and add with movement tags.
    canon = canonical_shared_names()
    id_map: dict[str, str] = {}
    for node in kg.nodes:
        graph_id, attrs = resolve_node_id(node.id, node.label, movement)
        id_map[node.id] = graph_id
        # Third defense against shared-layer fragmentation (design note 5): a shared node
        # whose name is neither a canonical vocab entry nor a known alias is genuinely new —
        # flag it so it can be reviewed into shared_vocab_v1.json instead of silently forking.
        if attrs.get("movement") == "shared" and attrs["name"] not in set(canon.get(node.label, [])):
            print(f"  [vocab-review] new shared {node.label}: '{attrs['name']}' (not in shared_vocab_v1.json)")
        if G.has_node(graph_id):
            G.nodes[graph_id].update(attrs)
        else:
            G.add_node(graph_id, **attrs)

    # Add Edges (endpoints remapped through id_map)
    for edge in kg.edges:
        source = id_map.get(edge.source, edge.source)
        target = id_map.get(edge.target, edge.target)
        if source == target:
            continue
        edge_exists = False
        existing_edge_data = G.get_edge_data(source, target)
        if existing_edge_data:
            if G.is_multigraph():
                for _, edge_data in existing_edge_data.items():
                    if isinstance(edge_data, dict) and edge_data.get('type') == edge.type:
                        edge_exists = True
                        break
            else:
                if isinstance(existing_edge_data, dict) and existing_edge_data.get('type') == edge.type:
                    edge_exists = True
        
        if not edge_exists:
            G.add_edge(source, target, type=edge.type)

    nx.write_graphml(G, graph_file)
    print(f"Graph updated and saved to {graph_file}")
    print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

def load_document(file_path):
    print(f"Loading document: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".html", ".htm"]:
        loader = BSHTMLLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        print(f"Unsupported file format: {ext}. Treating as text.")
        loader = TextLoader(file_path, encoding='utf-8')
    
    docs = loader.load()
    return docs

def main():
    parser = argparse.ArgumentParser(description="Extract Knowledge Graph from Documents.")
    parser.add_argument("filepath", type=str, nargs='?', help="Path to the document (PDF, HTML, TXT)")
    parser.add_argument("--graph-file", type=str, default=DEFAULT_GRAPH_FILE, help="Path to the output GraphML file")
    parser.add_argument("--movement", type=str, default="Squat",
                        help="Movement this document is about; scoped nodes are namespaced under it (e.g. Squat, Lunge, Push-up).")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    # api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY environment variable is not set.")
        print("Please set it before running.")
        return

    print("Initializing LLM pipeline (OpenRouter)...")
    llm = ChatOpenRouter(
        model="openrouter/auto",
        api_key=api_key,
        temperature=0
    )
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    structured_llm = llm.with_structured_output(KnowledgeGraph)

    # Steer the shared layer toward the controlled vocabulary so this movement's Cause/Cue/
    # Risk/QualityDimension nodes reuse existing shared ids instead of fragmenting them.
    canon = canonical_shared_names()
    vocab_lines = "\n".join(
        f"  {lbl}: {', '.join(names)}" for lbl, names in canon.items() if names
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert sports biomechanics researcher building an AQA-ready knowledge graph.\n"
                   f"The document concerns the '{args.movement}' movement. Extract only structured, high-value\n"
                   f"biomechanics knowledge that supports Action Quality Assessment for this movement.\n"
                   "Focus on the chain: Action -> Fault -> Phase -> EvidenceSignal -> Cause -> Risk -> Cue / QualityDimension.\n"
                   "Entities must be exactly one of: Action, Phase, Fault, EvidenceSignal, Cause, Risk, Cue, QualityDimension.\n"
                   "Relationships must be one of: HAS_PHASE, HAS_FAULT, OCCURS_IN_PHASE, INDICATED_BY, CAUSED_BY, INCREASES_RISK_OF, CORRECTED_BY, AFFECTS_QUALITY.\n"
                   "Use concise canonical names, prefer singular concepts, and avoid duplicate casing variants.\n"
                   "For the SHARED entity types (Cause, Cue, Risk, QualityDimension), PREFER these existing canonical\n"
                   "names verbatim whenever the concept matches one of them (they are shared across movements):\n"
                   f"{vocab_lines}\n"
                   "If the text is weakly structured, historical, promotional, or not actionable for AQA, extract only the strongest supported relationships."),
        ("human", "Extract the knowledge graph from the following text:\n\n{text}")
    ])

    chain = prompt | structured_llm

    # If a file is provided, process it. Otherwise, use sample text.
    if args.filepath:
        if not os.path.exists(args.filepath):
            print(f"File not found: {args.filepath}")
            return
            
        docs = load_document(args.filepath)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        print(f"Document split into {len(splits)} chunks.")

        for i, split in enumerate(splits):
            print(f"\nProcessing chunk {i+1} of {len(splits)}...")
            try:
                result = chain.invoke({"text": split.page_content})
                if result and result.nodes:
                    update_property_graph(result, graph_file=args.graph_file, movement=args.movement)
                    print(f"Successfully extracted {len(result.nodes)} nodes and {len(result.edges)} edges from chunk {i+1}.")
                else:
                    print(f"No relationships found in chunk {i+1}.")
            except Exception as e:
                print(f"An error occurred extracting from chunk {i+1}: {e}")
                
    else:
        print("No file provided.")

if __name__ == "__main__":
    main()
