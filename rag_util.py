# Description: This file contains utility functions to initialize the RAG pipeline and query the pipeline with a question.

import os, json
import torch
import sys, re
import logging
import ast
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
import openai
from openai import OpenAI

def logging_setup(log_file=".log", log_level=logging.INFO):
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # stream=sys.stdout, 
        level=log_level,
    )  # logging.DEBUG for more verbose output

    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    sys.stdout = open(log_file, "a")

def format_response(response):
    response = response.replace("\n", "")
    response = response.replace("), (", ") (")
    response = response.replace("),(", ") (")
    response = response.replace("'", "")
    response = response.replace(")(" , ") (")

    return response

def prepare_hdx(human_dx):
    logging.info(f"Count of human experts: {len(human_dx)}")

    # all_human_dx = []
    # for dx in human_dx:
    #     all_human_dx += dx
    # logging.info(f"Count of human expert diagnoses: {len(all_human_dx)}")
    # all_human_dx = list(set(all_human_dx))
    # logging.info(f"Count of unique human expert diagnoses: {len(all_human_dx)}")

    # all_human_dx = ', '.join(all_human_dx)

    all_human_dx = '. '.join(human_dx)
    
    return all_human_dx


def prepare_retrieval(material, retriever):

    if isinstance(material, list):
        logging.info(f"Retrieving for {len(material)} human expert diagnoses")
        material_for_retrieval = '. '.join(material)
    elif isinstance(material, str):
        logging.info(f"Retrieving for {material[:100]}...")
        material_for_retrieval = material

    retrieved_triples = retriever.retrieve(material_for_retrieval)
    retrieved_triples = list(dict.fromkeys(retrieved_triples[0].node.metadata['kg_rel_texts'])) # remove duplicates
    logging.info(f"Retrieved {len(retrieved_triples)} triples")

    retrieval = ", ".join(retrieved_triples) # join the triples with a comma into a single string

    return retrieval

def clean_structural_list(parsed_response, target_results):
    def fix_unmatched_quotes(text):
        single_quotes = text.count("'")
        double_quotes = text.count('"')
        if single_quotes % 2 != 0:
            text += "'"
        if double_quotes % 2 != 0:
            text += '"'
        return text

    cleaned_list = []
    try:
        parsed_response = parsed_response.strip()
        parsed_response = fix_unmatched_quotes(parsed_response)
        nested_list = ast.literal_eval(parsed_response)[target_results]
        for item in nested_list:
            if isinstance(item, list) and tuple(item) not in cleaned_list:
                cleaned_list.append(tuple(item))
            else:
                cleaned_list.append(str(item))
    except Exception as e:
        print(e)
        print("parsed_response: ", parsed_response)
    return cleaned_list
                

def extract_triple(answer, notebook=False, split_str1="Triples:", split_str2="Answer End"):
    answer = format_response(answer)
    try:
        if split_str1 in answer:
            answer = answer.split(split_str1)[1]
        if split_str2 in answer:
            answer = answer.split(split_str2)[0]
        else:
            pass
        return answer.strip().replace("\n", "") if not notebook else answer.strip()
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':
    pass

    # TODO the following code to test the functions from this file
    # from llama_index.query_engine import RetrieverQueryEngine
    # from llama_index.retrievers import KnowledgeGraphRAGRetriever

    # llm, service_context = init_llm_service_context()
    # storage_context = init_kg_storage_context()

    # graph_rag_retriever = KnowledgeGraphRAGRetriever(
    #     storage_context=storage_context,
    #     service_context=service_context,
    #     llm=llm,
    #     verbose=True,
    # )

    # query_engine = RetrieverQueryEngine.from_args(
    #     graph_rag_retriever, service_context=service_context
    # )
    # logging.info("Query engine created")

    # response = query_engine.query(
    #     "Generate triplets of disease which relate to cyclophosphamide with the format of (subject ; has adverse effect ; object), \
    #         \
    #         For example, (cyclophosphamide ; has adverse effect ; hemorrhagic cystitis)",
    #     )
    # # display(Markdown(f"<b>{response}</b>"))
    # print(response.response)