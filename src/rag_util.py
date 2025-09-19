# Description: This file contains utility functions to initialize the RAG pipeline and query the pipeline with a question.

import os, json
import torch
import sys, re
import logging
import ast
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") if os.getenv("DEEPSEEK_API_KEY") else ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") if os.getenv("NVIDIA_API_KEY") else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

EVAL_DATA = {
    "BC5CDR": "data/BioNEL_datasets/BC5CDR_gold_all.json",
    "NCBI": "data/BioNEL_datasets/NCBIdevelopset.json",
    "MIMICIV-chunk": "data/mimiciv/eval/notes_concepts_chunked.json",
    "MIMICIV": "data/mimiciv/eval/notes_concepts_all.json",
    "pubmed": "data/pubmed_eval_datasets/2023_selected_limit_50.json",
    # "humandx": "data/humandx_data/humandx_diagnosis.json"
    "humandx": "data/humandx_data/humandx_diagnosis_random_solvers_id_SPARQL.json"
}

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
        if single_quotes % 2 != 0 and text[-1] == "'":
            text = text[:-1]
        if double_quotes % 2 != 0 and text[-1] == '"':
            text = text[:-1]
        return text

    def fix_incomplete_json(text):
        # If the text ends with a comma, remove it
        if text.strip().endswith(','):
            text = text.strip()[:-1]
        # If the text ends with an incomplete string, remove it
        if text.strip().endswith('"') or text.strip().endswith("'"):
            text = text.strip()[:-1]
        # If the text ends with an incomplete list, complete it
        if text.strip().endswith('['):
            text = text.strip()[:-1]
        # If the text ends with an incomplete dictionary, complete it
        if text.strip().endswith('{'):
            text = text.strip()[:-1]
        # Add closing brackets if needed
        if text.count('[') > text.count(']'):
            text += ']' * (text.count('[') - text.count(']'))
        if text.count('{') > text.count('}'):
            text += '}' * (text.count('{') - text.count('}'))
        return text

    def remove_last_element_and_complete(text):
        # Find the last comma before the end of the list
        last_comma = text.rfind(',')
        if last_comma != -1:
            # Find the start of the last element
            start_of_last = text[:last_comma].rfind('"') + 1
            if start_of_last > 0:
                # Remove the last element and complete the JSON
                text = text[:start_of_last] + ']}'
        return text

    cleaned_list = []
    try:
        parsed_response = parsed_response.strip()
        parsed_response = fix_unmatched_quotes(parsed_response)
        
        # If the JSON is incomplete, remove the last element and complete it
        if not parsed_response.endswith('"}') and not parsed_response.endswith('"]}'):
            parsed_response = remove_last_element_and_complete(parsed_response)
        
        parsed_response = fix_incomplete_json(parsed_response)
        nested_list = ast.literal_eval(parsed_response)[target_results]
        for item in nested_list:
            if isinstance(item, list) and tuple(item) not in cleaned_list:
                cleaned_list.append(tuple(item))
            else:
                cleaned_list.append(str(item))
    except Exception as e:
        print(e)
        print("parsed_response: ", parsed_response)
        if parsed_response.startswith(f"{target_results}"):
            temp_list = parsed_response.split("Concepts:")[1].strip().split("\n")[0].strip()
            cleaned_list = [item.strip() for item in temp_list.strip('[]').split(",") if item.strip()]
    
    if len(cleaned_list) > 0:
        cleaned_list = list(set(cleaned_list)) # remove duplicates
    
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


# Define a function to clean the response
def clean_response(response):

    def remove_repeated_pairs_and_unbalanced_braces(text):
        # Find all pairs of the format (a; b)
        pattern = r'\([^()]*;[^()]*\)'
        pairs = re.findall(pattern, text)
        
        seen = set()
        unique_pairs = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)
        for pair in pairs:
            text = text.replace(pair, '', 1)
        
        cleaned_text = ' '.join(unique_pairs)
        cleaned_text += ' ' + text
        
        # Function to remove unbalanced braces and the text within them
        def remove_unbalanced_braces(text):
            stack = []
            balanced = [True] * len(text)  # Marks whether each character is part of a balanced segment
            
            matching_brace = {')': '(', ']': '[', '}': '{'}
            opening_brace = set(matching_brace.values())
            closing_brace = set(matching_brace.keys())
            
            for i, char in enumerate(text):
                if char in opening_brace:
                    stack.append((char, i))
                elif char in closing_brace:
                    if stack and stack[-1][0] == matching_brace[char]:
                        stack.pop()
                    else:
                        balanced[i] = False
            
            # If there are unmatched opening braces left in the stack, mark them as unbalanced
            while stack:
                _, pos = stack.pop()
                balanced[pos] = False
            
            # Build the cleaned text
            cleaned_text = []
            skip = False
            for i, char in enumerate(text):
                if char in opening_brace or char in closing_brace:
                    if not balanced[i]:
                        skip = True
                    if balanced[i]:
                        skip = False
                if not skip:
                    cleaned_text.append(char)
                if char in closing_brace and not balanced[i]:
                    skip = False  # Stop skipping after an unbalanced closing brace
            
            return ''.join(cleaned_text)

        cleaned_text = remove_unbalanced_braces(cleaned_text)

        return cleaned_text.strip()
    
    def remove_punts(text):
        text = text.replace("'", "")
        text = text.replace('"', "")
        if text.find(";") != -1:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ";")
        return text
    
    def remove_numerical_order_list(text):
        pattern = r'\b\d+\.\s*'
        cleaned_text = re.sub(pattern, '', text)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    def remove_numerical_order_list2(text):
        pattern = r'\b\d+\)\s*'
        cleaned_text = re.sub(pattern, '', text)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    keywords = ["Note:", "Explanation:", "Prompt:", "Context:", "Context", "Queries:", "Query:", "Here is", "Code:", "---"]
    # response_list = response.split("\n")
    for keyword in keywords:
        if keyword in response:
            response = response.split(keyword)[0]
    response = remove_punts(response.replace("\n", "").strip())
    response = remove_repeated_pairs_and_unbalanced_braces(response)
    response = remove_numerical_order_list(response)
    response = remove_numerical_order_list2(response)
    # response_list = response.replace("\n", "").split(") (")
    # response = " | ".join([r.strip() for r in response_list if r])
    return response




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