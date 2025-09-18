# RAG_Demo

A comprehensive Retrieval-Augmented Generation (RAG) system for medical and healthcare applications, supporting various LLM models and embedding techniques.

## Overview

This project implements a RAG system specifically designed for medical and healthcare applications. It supports multiple Large Language Models (LLMs) and embedding models, with a focus on extracting and processing medical concepts, relationships, and entities.

## Features

- Support for multiple LLM models:
  - Mistral Small
  - Llama 3.1 8B
  - GPT-4
  - NuExtract models
  - Custom models through HuggingFace

- Multiple embedding models:
  - BGE-M3
  - MiniLM variants
  - HiT-MiniLM-L12-SnomedCT

- Specialized medical concept extraction and processing:
  - SNOMED CT concept extraction
  - Medical entity recognition
  - Relationship extraction
  - Knowledge graph integration

## Project Structure

```
RAG_Demo/
├── data/                  # Data storage directory
├── index/                 # Index storage for vector and knowledge graphs
├── llm/                   # LLM model storage
├── logs/                  # Log files
├── rag_moduler.py         # Core RAG functionality and LLM pipeline
├── rag_prompt_template.py # Prompt templates for various tasks
├── rag_util.py           # Utility functions
├── llm_extraction.py     # LLM-based extraction utilities
├── rag_extraction.py     # RAG-specific extraction utilities
├── rag_index_builder_csv.py # Index building utilities
└── rag_playground.ipynb  # Interactive notebook for experimentation
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## Usage

### Basic RAG Pipeline

```python
from rag_moduler import init_llm_service_context, init_kg_storage_context, init_rag_pipeline

# Initialize LLM service
llm = init_llm_service_context(
    llm_model_name="mistralsmall",
    embed_model_name="bgem3"
)

# Initialize knowledge graph storage
kg_index = init_kg_storage_context(llm)

# Initialize RAG pipeline
rag_pipeline = init_rag_pipeline(
    kg_index,
    similarity_top_k=3,
    retriever_mode="hybrid"
)
```

### Medical Concept Extraction

```python
from rag_prompt_template import snomed_extraction_prompt

# Extract SNOMED CT concepts
concepts = llm_entity_extractor(
    text="Patient presented with acute myocardial infarction",
    pipe=llm_pipeline,
    using_extractor="nuparser"
)
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