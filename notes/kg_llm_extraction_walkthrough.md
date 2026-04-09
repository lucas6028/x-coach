# Knowledge Graph Schema & LLM Extraction Pipeline

## Changes Made
1. **Designed the KG Schema**: Modeled the sports biomechanics causal chain based on [project-overview.md](file:///c:/Users/Hao/Code/my-project/x-coach/project-overview.md) and `計劃書.pdf`. The core schema includes:
   * **Nodes (Entities):** `Action`, `Error`, `Anatomy`, `Risk`, `Correction`
   * **Edges (Relationships):** `HAS_POTENTIAL_ERROR`, `CAUSED_BY_WEAKNESS_IN`, `INCREASES_RISK_OF`, `CORRECTED_BY`

2. **Implemented Extraction Pipeline**: Wrote [extract_kg.py](file:///c:/Users/Hao/Code/my-project/x-coach/extract_kg.py) using the LangChain `with_structured_output` API and Pydantic models. 
3. **Switched to Gemini**: Migrated the extraction pipeline to **Google Gemini (gemini-2.5-flash)** using `langchain-google-genai`.
4. **Persistent Labeled Property Graph (LPG)**: Added `networkx` integration to merge the extracted `KnowledgeGraph` into a local LPG (`sports_kg.graphml`). This allows incremental graph expansion as multiple documents are passed through the pipeline.
5. **Multi-Format Document Loaders**: Integrated `langchain_community.document_loaders` (`PyPDFLoader`, `BSHTMLLoader`, `TextLoader`) and `RecursiveCharacterTextSplitter` to handle actual file inputs (PDF, HTML, Text). 
6. **Installed Dependencies**: Set up `pydantic`, `langchain`, `langchain-google-genai`, `networkx`, `pypdf`, and `beautifulsoup4` into the virtual environment.

## Validation Results
Ran the Python script locally to ensure there are no compilation errors or missing dependencies. The pipeline is successfully initialized and awaiting a `GOOGLE_API_KEY` to perform LLM inference on the documents. 

To run the pipeline and generate `kg_extracted.json` & update the LPG `sports_kg.graphml`, set your API key environment variable and run:
```powershell
$env:GOOGLE_API_KEY="your-gemini-key"
python src/kg/extract_kg.py [path_to_document]
```

