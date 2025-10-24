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
├── requirements_minimal.txt    # The top-level necessary dependencies
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
    ├── llm_factory.py          # Modular factory for building LLMs and embedding models
    ├── rag_auxiliaries         # Auxiliary LLM utilities (extractor, parser, generator)
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

2. (Recommended) Create a Python 3.10 environment:
```bash
conda create -n hacid_rag python=3.10
conda activate hacid_rag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### requirements.txt vs requirements_minimal.txt

- **requirements.txt**: Contains all packages (including sub-dependencies and system/conda packages) from a full environment export. Use this for exact environment replication.
- **requirements_minimal.txt**: Contains only the top-level, necessary Python packages. All other dependencies will be installed automatically. Use this for a cleaner, more portable, and easier-to-maintain setup.

**Recommendation:**
- Use `requirements.txt` for legacy or debugging purposes.
- Use `requirements_minimal.txt` for new installations or lightweight environments.

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

## Recent Updates

### New Modular Architecture (v0.2)

The codebase has been refactored to improve modularity and maintainability:

#### New File: `llm_factory.py`
A dedicated factory module for building and managing LLMs and embedding models:
- **`build_hf_transformers()`**: Explicitly load HuggingFace models with fine-grained control over quantization, device mapping, and memory management
- **`wrap_hf_llm()`**: Wrap Transformers models into LlamaIndex `HuggingFaceLLM` objects
- **`build_llm_by_name()`**: Factory method supporting multiple LLM types (HuggingFace, OpenAI, DeepSeek)
- **`build_embed_model()`**: Create embedding models with consistent interface
- **`configure_global_settings()`**: Set global LLM and embedding model settings
- **`release_model()`**: Properly release models and free GPU memory

#### New File: `rag_auxiliaries`
Auxiliary LLM utilities extracted from the original `rag_moduler.py` for better code organization:
- **`init_llm_pipeline()`**: Initialize LLM pipeline for extractors, parsers, and generators
- **`llm_entity_extractor()`**: Extract entities from text using LLM
- **`llm_parser()`**: Parse RAG or LLM extractor outputs
- **`llm_generation()`**: Generate responses from LLM models

#### Modified: `rag_moduler.py`
Core RAG functionality now leverages the new factory pattern and has been streamlined:
- Imports and uses functions from `llm_factory.py` for all model initialization
- Moved auxiliary LLM functions (extractors, parsers, generators) to `rag_auxiliaries`
- Simplified `init_llm_service_context()` by delegating model loading to factory functions
- Improved `init_kg_storage_context()` with better embedding model handling
- Enhanced `CustomKGTableRetriever` with Ollama-based semantic similarity computation
- Added `swap_llm_and_rebuild_engine()` for dynamic LLM switching (experimental)

These changes provide:
- **Better separation of concerns**: Model management is isolated from RAG logic
- **Improved memory management**: Explicit control over model loading and release
- **Enhanced flexibility**: Easier to add new LLM providers or modify model configurations
- **Cleaner code**: Reduced duplication and improved maintainability

## Configuration

The system supports various configurations through the `llm_factory.py` and `rag_moduler.py` files:

- LLM model selection (HuggingFace, OpenAI, DeepSeek)
- Embedding model selection
- Quantization settings (4-bit, 8-bit)
- Context window size
- Temperature and sampling parameters
- Device mapping and memory optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.