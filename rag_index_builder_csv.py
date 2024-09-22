import os
import json
import logging
import sys
from transformers import BitsAndBytesConfig
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from llama_index.core import KnowledgeGraphIndex, ServiceContext, SimpleDirectoryReader, download_loader, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
from IPython.display import Markdown, display
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from rag_util import *

import pandas as pd


def extract_triples(text):
    triple_list = []
    # Extract the values of the desired columns for each row
    # result = df[['first', 'second', 'third']].values.tolist()
    result = triple_df[['subj', 'prop', 'obj']].values.tolist()

    # Print the result
    for row in result:
        triple_list.append((row[0], row[1], row[2]))
    
    return triple_list


def build_kg_index(llm_model_name, 
                    tokenizer_name, 
                    embed_model_name,
                    index_persist_dir="index/snomed_dataset_nodoc", 
                    document_input_dir=None):
    if document_input_dir is None:
        document_input_dir = "data/empty"
    reader = SimpleDirectoryReader(input_dir=document_input_dir, exclude=["*.target"])
    documents = reader.load_data()

    logging.info("Loaded {} documents: {}".format(len(documents), documents[0]))

    triple_list = extract_triples('')
    print(triple_list[:5])
    print(len(triple_list))
    logging.info("Triple amount: " + str(len(triple_list)))

    llm, service_context = init_llm_service_context(
        llm_model_name=llm_model_name, 
        tokenizer_name=tokenizer_name, 
        embed_model_name=embed_model_name
    )

    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store = graph_store)
    
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        # max_triplets_per_chunk=10,
        service_context=service_context,
        include_embeddings=True,
        kg_triplet_extract_fn=extract_triples,
    )

    logging.info("KG Indexing complete")

    kg_index.storage_context.persist(persist_dir=index_persist_dir)
    logging.info("KG Indexing persisted")


if __name__ == "__main__":

    using_llm = "commandr"
    using_embed = "hitsnomed"

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM[using_llm],
        "embed_model_name": EMBED_MODEL[using_embed],
        "index_persist_dir": f"index/snomed_dataset_nodoc_{using_llm}_{using_embed}",
        "knowledge_graph_dataset_dir": "data/snomed/dataset.csv",
    }

    logging_setup(log_file="logs/KG_RAG_simple_builder_csv.log", log_level=logging.INFO)

    triple_df = pd.read_csv(PARAMETERS["knowledge_graph_dataset_dir"], sep=';')
    
    build_kg_index(llm_model_name=PARAMETERS["llm_model_name"], 
                    tokenizer_name=PARAMETERS["tokenizer_name"], 
                    embed_model_name=PARAMETERS["embed_model_name"],
                    index_persist_dir=PARAMETERS["index_persist_dir"])
