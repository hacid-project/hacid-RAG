# Notebooks

This directory contains Jupyter notebooks for experimentation, testing, and demonstration of the HACID_RAG system's capabilities.

## Overview

The notebooks provide hands-on examples and testing environments for various RAG (Retrieval-Augmented Generation) functionalities, focusing on medical and healthcare applications. Each notebook demonstrates different aspects of the system, from basic pipeline setup to advanced feature testing.

## Available Notebooks

### `rag_playground.ipynb`
**Main interactive playground for comprehensive RAG system testing**

**Pipeline Features:**
- Complete RAG pipeline initialization with configurable LLM and embedding models
- Knowledge graph retrieval with hybrid similarity search
- Medical text processing for multiple NLP tasks
- Structured output parsing and validation

**Key Examples:**

1. **Pure Retrieval Testing**
   - Input: Medical case vignette (e.g., "40 year old female presenting with chest pain...")
   - Output: Retrieved knowledge graph triples related to symptoms and conditions

2. **Question-Answering**
   - Input: "What is the type of the concept 'Biliary pancreatitis'?"
   - Output: SNOMED CT concept type classification

3. **Medical Concept Extraction**
   - Input: Discharge note text with medical history
   - Output: Extracted SNOMED CT concepts categorized by type (finding, disorder, procedure, etc.)

4. **Entity-Type Pair Extraction**
   - Input: Medical abstract about drug effects
   - Output: Structured pairs like "(depression ; Disorder)" and "(methyl dopa ; Substance)"

5. **Triple Extraction**
   - Input: Research abstract about surgical procedures
   - Output: Medical knowledge triples in format "(concept1 ; relation ; concept2)"

6. **Medical Diagnosis Generation**
   - Input: Clinical case vignette with symptoms
   - Output: Ranked differential diagnoses based on probability

### `retriever_test.ipynb`
**Focused testing and comparison of retrieval components**

**Pipeline Features:**
- Systematic evaluation of default vs custom retrievers
- Performance comparison with different similarity thresholds
- Knowledge graph exploration with configurable depth parameters

**Key Examples:**

1. **Default Retriever Testing**
   - Input: Complex medical text about cerebrospinal fluid leakage
   - Output: Top-K most relevant triples from knowledge graph

2. **Custom Retriever with Enhanced Features**
   - Input: Same medical text as above
   - Output: Filtered and ranked results with additional semantic scoring
   - Features: Relation-specific filtering, similarity thresholds, hybrid scoring

### `RAG_format_enforcer.ipynb`
**Triple extraction with structured output enforcement**

**Pipeline Features:**
- Medical abstract processing for SNOMED CT knowledge extraction
- Format enforcement to ensure consistent triple structure
- Hybrid extraction engine combining RAG retrieval with structured generation

**Key Examples:**

1. **Medical Abstract Processing**
   - Input: Research abstracts about surgical procedures (thyroidectomy, cardiac conditions)
   - Output: Structured triples with enforced format: `[(subject; predicate; object), ...]`

2. **Format-Enforced Extraction**
   - Input: "Traditional thyroidectomy results in neck scar..."
   - Output: Validated triples like "(thyroidectomy; associated with; neck scar)" with guaranteed formatting

### `RAG_grammarllm.ipynb`
**Grammar-constrained generation for reliable structured outputs**

**Pipeline Features:**
- Context-aware triple extraction using grammar-based constraints
- Integration with knowledge graph retrieval for enhanced accuracy
- Systematic evaluation on PubMed datasets with predefined relation vocabularies

**Key Examples:**

1. **Grammar-Constrained Triple Generation**
   - Input: Medical text about clinical procedures and findings
   - Output: Syntactically valid triples conforming to predefined grammar rules
   - Format: Enforced structure `[(SUBJECT; PREDICATE; OBJECT), ...]`

2. **Dataset Evaluation Pipeline**
   - Input: PubMed research abstracts from evaluation datasets
   - Output: Batch-processed triples with quality metrics and validation
   - Features: Automated evaluation, performance tracking, constrained vocabulary

## Getting Started

1. Ensure you have completed the main repository setup (see main README.md)
2. Download required models and knowledge graph indices as described in the main README
3. Start with `rag_playground.ipynb` for a comprehensive introduction
4. Use specific notebooks based on your testing needs:
   - For retrieval testing: `retriever_test.ipynb`
   - For format enforcement: `RAG_format_enforcer.ipynb`
   - For grammar constraints: `RAG_grammarllm.ipynb`

## Prerequisites

- Python environment with required dependencies installed
- Access to LLM models and embedding models
- Knowledge graph indices downloaded and placed in appropriate directories
- GPU recommended for optimal performance

## Usage Tips

- Start with small test cases before processing large datasets
- Monitor memory usage when working with large knowledge graphs
- Adjust similarity thresholds and retrieval parameters based on your use case
- Use the logging features to track performance and debug issues

## Contributing

When adding new notebooks:
- Follow the existing naming conventions
- Include comprehensive documentation and examples
- Add appropriate error handling and validation
- Update this README with new notebook descriptions