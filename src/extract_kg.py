import os
import sys
import json
import argparse
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

class Node(BaseModel):
    id: str = Field(description="Unique identifier for the node (e.g., 'Squat', 'Knee Valgus')")
    label: str = Field(description="The type of the entity (Action, Error, Anatomy, Risk, Correction)")

class Edge(BaseModel):
    source: str = Field(description="The id of the source node")
    target: str = Field(description="The id of the target node")
    type: str = Field(description="The relationship type: HAS_POTENTIAL_ERROR, CAUSED_BY_WEAKNESS_IN, INCREASES_RISK_OF, CORRECTED_BY")

class KnowledgeGraph(BaseModel):
    """A knowledge graph consisting of nodes and edges representing sports biomechanics."""
    nodes: List[Node] = Field(description="List of entities (nodes) in the knowledge graph")
    edges: List[Edge] = Field(description="List of relationships (edges) in the knowledge graph")

def update_property_graph(kg: KnowledgeGraph, graph_file="data/kg/sports_kg.graphml"):
    """Adds the extracted Knowledge Graph data into a persistent NetworkX GraphML file."""
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

    # Add Nodes
    for node in kg.nodes:
        if G.has_node(node.id):
            G.nodes[node.id]['label'] = node.label
        else:
            G.add_node(node.id, label=node.label)

    # Add Edges
    for edge in kg.edges:
        edge_exists = False
        existing_edge_data = G.get_edge_data(edge.source, edge.target)
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
            G.add_edge(edge.source, edge.target, type=edge.type)

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert sports biomechanics researcher. Extract a knowledge graph from the given text.\n"
                   "Focus on the structured relationship chain: Action -> Error -> Anatomy/Cause -> Risk -> Correction.\n"
                   "Entities should be exactly one of: Action (e.g., Squat), Error (e.g., Knee Valgus), Anatomy (e.g., Gluteus Medius), Risk (e.g., ACL injury), Correction (e.g., Push knees outward).\n"
                   "Relationships should be one of: HAS_POTENTIAL_ERROR, CAUSED_BY_WEAKNESS_IN, INCREASES_RISK_OF, CORRECTED_BY."),
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
                    update_property_graph(result)
                    print(f"Successfully extracted {len(result.nodes)} nodes and {len(result.edges)} edges from chunk {i+1}.")
                else:
                    print(f"No relationships found in chunk {i+1}.")
            except Exception as e:
                print(f"An error occurred extracting from chunk {i+1}: {e}")
                
    else:
        print("No file provided. Running on sample text...")
        # sample_text = """
        # 在深蹲(Squat)過程中，常見的錯誤是膝蓋內扣(Knee Valgus)。這通常是因為臀中肌無力(Gluteus Medius weakness)所導致。
        # 如果不加以修正，膝蓋內扣會增加前十字韌帶(ACL)受傷的風險。
        # 為了修正這個錯誤，教練可以建議學員在膝蓋套上彈力帶，並在下蹲時想像將彈力帶撐開(Push knees outward with resistance band)。
        # """
        # try:
        #     result = chain.invoke({"text": sample_text})
        #     output_file = "kg_extracted.json"
        #     with open(output_file, "w", encoding="utf-8") as f:
        #         json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            
        #     print(f"Extracted KG saved to {output_file}")
        #     # Use model_dump instead of dict for Pydantic V2
        #     print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
            
        #     update_property_graph(result)
        # except Exception as e:
        #     print(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    main()
