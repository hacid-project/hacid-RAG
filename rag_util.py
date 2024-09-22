# Description: This file contains utility functions to initialize the RAG pipeline and query the pipeline with a question.

import os
import torch
import sys, re
import logging
from transformers import BitsAndBytesConfig

from llama_index.core import KnowledgeGraphIndex, ServiceContext, SimpleDirectoryReader, download_loader, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM, HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
from IPython.display import Markdown, display
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.core.embeddings.openai import OpenAIEmbedding


LLM = {
    "commandr": "CohereForAI/c4ai-command-r-v01-4bit",
    "zephyralpha": "HuggingFaceH4/zephyr-7b-alpha",
    "zephyrbeta": "HuggingFaceH4/zephyr-7b-beta",
}

EMBED_MODEL = {
    "minilml6v2": "sentence-transformers/all-MiniLM-L6-v2",
    "minilml12v2": "sentence-transformers/all-MiniLM-L12-v2",
    "hitsnomed": "Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT"
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


def extract_triple(answer, notebook=False, split_str1="Triples:", split_str2="Answer End"):
    split_str1 = "Triples:"
    split_str2 = "Answer End"

    try:
        if split_str2 in answer:
            answer = answer.split(split_str2)[0]
        if split_str1 in answer:
            answer = answer.split(split_str1)[1]
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

    keywords = ["Note:", "Explanation:", "Prompt:", "Context:", "Context", "Queries:", "Query:", "Answer:", "Here is", "Code:", "---"]
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


# Load the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == 'system':
            prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"<|assistant|>\n{message.content}</s>\n"

        # ensure we start with a system prompt, insert blank if needed
        if not prompt.startswith("<|system|>\n"):
            prompt = "<|system|>\n</s>\n" + prompt

        # add final assistant prompt
        prompt = prompt + "<|assistant|>\n"

    return prompt


def init_llm_service_context(llm_model_name="HuggingFaceH4/zephyr-7b-alpha", 
                             tokenizer_name="HuggingFaceH4/zephyr-7b-alpha", 
                             embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                             quantization_config=quantization_config, 
                             token=None, 
                             output_parser=None,):
    if "gpt" in llm_model_name:
        llm = OpenAI(model=llm_model_name, api_key=os.environ["OPENAI_API_KEY"], temperature=0.7)
        embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    else:
        llm = HuggingFaceLLM(
            model_name=llm_model_name, 
            tokenizer_name=tokenizer_name, 
            context_window=2048,
            max_new_tokens=256,
            model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
            # tokenizer_kwargs={"trust_remote_code": True,},
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True},
            # messages_to_prompt=messages_to_prompt,
            device_map="auto",
            output_parser=output_parser,
        )

        embed_model = HuggingFaceEmbeddings(model_name=embed_model_name) 
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model=embed_model)
    logging.info(f"LLM loaded: {llm.model_name}" if "gpt" not in llm_model_name else f"LLM loaded: {llm.model}")
    logging.info(f"embed_model loaded: {embed_model.model_name}")
    logging.info("Service context created")

    return llm, service_context


def init_kg_storage_context(storage_dir="data/index/ade_full_graph_doc"):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    logging.info(f"KG storage: {storage_dir}")
    logging.info("KG Storage context loaded")

    return storage_context


def init_vector_storage_context(storage_dir="data/index/ade_full_vector"):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    logging.info(f"Vector storage: {storage_dir}")
    logging.info("Vector storage context loaded")

    return storage_context


def init_rag_pipeline(llm, 
                      service_context, 
                      storage_context, 
                      include_text=False, 
                      similarity_top_k=3, 
                      graph_store_query_depth=2, 
                      verbose=False
                      ):
    kg_index = load_index_from_storage(
        storage_context=storage_context,
        service_context=service_context,
        llm=llm,
    )

    kg_query_engine = kg_index.as_query_engine(
        # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
        include_text=include_text,
        retriever_mode="hybrid",
        embedding_mode="hybrid",
        response_mode="compact",
        similarity_top_k=similarity_top_k,
        graph_store_query_depth=graph_store_query_depth,
        explore_global_knowledge=True,
        verbose=verbose,
    )

    logging.info("RAG pipeline created")
    return kg_query_engine


if __name__ == '__main__':
    from llama_index.query_engine import RetrieverQueryEngine
    from llama_index.retrievers import KnowledgeGraphRAGRetriever

    llm, service_context = init_llm_service_context()
    storage_context = init_kg_storage_context()

    graph_rag_retriever = KnowledgeGraphRAGRetriever(
        storage_context=storage_context,
        service_context=service_context,
        llm=llm,
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(
        graph_rag_retriever, service_context=service_context
    )
    logging.info("Query engine created")

    response = query_engine.query(
        "Generate triplets of disease which relate to cyclophosphamide with the format of (subject ; has adverse effect ; object), \
            \
            For example, (cyclophosphamide ; has adverse effect ; hemorrhagic cystitis)",
        )
    # display(Markdown(f"<b>{response}</b>"))
    print(response.response)