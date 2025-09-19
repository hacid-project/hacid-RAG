# HACID_RAG

A comprehensive Retrieval-Augmented Generation (RAG) system for medical and healthcare applications, supporting various LLM models and embedding techniques.

> **Note:** The current repository version is for test usage and it's under update.

## Overview

This project implements a RAG system specifically designed for medical and healthcare applications. It supports multiple Large Language Models (LLMs) and embedding models, with a focus on extracting and processing medical concepts, relationships, and entities.

## Features

- Specialized medical concept extraction and processing:
  - SNOMED CT concept extraction
  - Medical entity recognition
  - Relationship extraction
  - Knowledge graph integration

## Project Structure

```
HACID_RAG/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── .env.template               # Environment variables template
├── data/                       # Data storage directory
│   ├── BC5CDR/                 # BC5CDR dataset
│   ├── mimiciv/                # MIMIC-IV dataset
│   └── pubmed/                 # PubMed dataset
├── notebooks/                  # Jupyter notebooks for experimentation and testing
│   ├── rag_playground.ipynb    # Main playground for RAG system testing
│   ├── retriever_test.ipynb    # Retriever functionality testing
│   ├── RAG_format_enforcer.ipynb # Triple extraction with format enforcement
│   ├── RAG_grammarllm.ipynb    # Grammar-constrained triple extraction
│   └── grammarllm/             # Grammar LLM utilities
│       └── temp/
├── scripts/                    # Utility scripts (currently empty)
└── src/                        # Source code directory
    ├── rag_moduler.py          # Core RAG functionality and LLM pipeline
    ├── rag_prompt_template.py  # Prompt templates for various tasks
    ├── rag_util.py            # Utility functions
    ├── llm_extraction.py      # LLM-based extraction utilities
    ├── rag_extraction.py      # RAG-specific extraction utilities
    ├── rag_index_builder_csv.py # Index building utilities
    ├── rag_llm_series_extraction.py # Series extraction utilities
    ├── rag_run.sh             # Shell script for running RAG
    ├── format_enforcer_wrapper.py # Format enforcement wrapper
    ├── grammar_llm_utils.py   # Grammar LLM utilities
    ├── deepseek_llm.py        # DeepSeek LLM integration
    ├── grammarllm/            # Grammar LLM package
    │   ├── README_PACKAGE.md  # Package documentation
    │   ├── __init__.py        # Package initialization
    │   ├── generate_with_constraints.py # Constrained generation
    │   ├── modules/           # Grammar modules
    │   ├── scripts/           # Grammar scripts
    │   └── utils/             # Grammar utilities
    └── utils/                 # General utilities
        └── deepseek_llm.py    # DeepSeek LLM utilities
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for experimentation, testing, and demonstration of the RAG system's capabilities. For detailed descriptions of each notebook's features, pipeline, and examples, see [`notebooks/README.md`](notebooks/README.md).

### Available Notebooks:
- **`rag_playground.ipynb`** - Main interactive playground for comprehensive RAG system testing
- **`retriever_test.ipynb`** - Focused testing and comparison of retrieval components
- **`RAG_format_enforcer.ipynb`** - Triple extraction with structured output enforcement
- **`RAG_grammarllm.ipynb`** - Grammar-constrained generation for reliable structured outputs

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Additional Installation Step

Ensure you have the `ollama` package installed, as it is required for embedding computations in the Custom RAG retriever in the [src/rag_moduler.py](https://github.com/hacid-project/hacid-RAG/blob/master/src/rag_moduler.py). You can install it using the following command:

```bash
pip install ollama
```

## Quick Usage

1. Download the .zip file of the indexed knowledge graph [here](https://drive.google.com/drive/folders/1FuU0hpjdNEFad9uB6rIP38F5Qygr7Dme).
   - **snomed_all_dataset_nodoc_hitsnomed.zip:** The indices of full knowledge graph embedded by [hitsnomed](https://huggingface.co/Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT)
   - **snomed_all_dataset_nodoc_minilml12v2.zip:** The indices of full knowledge graph embedded by [minilml12v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
   - **snomed_partial_dataset_nodoc_minilml6v2.zip:** The indices of **partial** knowledge graph embedded by [minilml6v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

2. Download the LLM and embedding model (recommendations: [mistral-small](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) and [hitsnomed](https://huggingface.co/Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT) ). Place them in a new folder and check the paths in the [src/rag_moduler.py](https://github.com/hacid-project/hacid-RAG/blob/master/src/rag_moduler.py)

   `$ LLM = {}` and 
   `EMBED_MODEL = {}`

3. Unzip the file and put it into the "index" folder. (If there is no "index" folder, create a new one under the root directory.)

**Note:** The folders "logs", "results", and "llm" need to be created manually if they do not exist.

### Basic RAG Pipeline

- This section shows how to initialize and run the core RAG pipeline. Implementation details are in `src/rag_moduler.py`, and a live example is available in `notebooks/rag_playground.ipynb`.

1. Initialize the LLM service context with your chosen LLM and embedding model:
```python
from rag_moduler import init_llm_service_context
llm = init_llm_service_context(
    llm_model_name="mistralsmall",
    embed_model_name="bgem3"
)
```

2. Load the knowledge graph index (create or load an existing index directory):
```python
from rag_moduler import init_kg_storage_context
kg_index = init_kg_storage_context(
    llm,
    storage_dir="path/to/your/index"
)
```

3. Build the RAG pipeline by combining the LLM and KG index into a query engine:
```python
from rag_moduler import init_rag_pipeline
rag_pipeline = init_rag_pipeline(
    kg_index,
    similarity_top_k=3,
    retriever_mode="hybrid"
)
```

4. Query the pipeline and inspect the response:
```python
response = rag_pipeline.query("What medical concepts are associated with acute myocardial infarction?")
print(response)
```

## Supported Tasks

1. **Medical Concept Extraction**
   - SNOMED CT concept extraction
   - BC5CDR entity-type extraction
   - NCBI disease entity extraction
   - MIMIC-IV SNOMED entity extraction

2. **Relationship Extraction**
   - SNOMED CT relationship extraction
   - Medical concept relationship mapping

3. **Knowledge Graph Integration**
   - Vector storage
   - Knowledge graph storage
   - Hybrid retrieval

## Configuration

The system supports various configurations through the `rag_moduler.py` file:

- LLM model selection
- Embedding model selection
- Quantization settings
- Context window size
- Temperature and sampling parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.