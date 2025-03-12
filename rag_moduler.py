import os, json
import torch
import sys, re
import logging
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
import openai
from openai import OpenAI
OPENAI_API_KEY = ""
DEEPSEEK_API_KEY = ""
os.environ["OPENAI_API_KEY"] = ""

from llama_index.core import KnowledgeGraphIndex, Settings, ServiceContext, SimpleDirectoryReader, download_loader, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
from IPython.display import Markdown, display
from langchain_huggingface import HuggingFaceEmbeddings
from deepseek_llm import DeepSeek
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.core.embeddings.openai import OpenAIEmbedding

LLM = {
    "mistralsmall": "llm/Mistral-Small-Instruct-2409",
    "llama3.18b": "llm/Llama-3.1-8B-Instruct",
    "gpt-4o-mini": "gpt-4o-mini",
    "nuparser": "llm/NuExtract-1.5-tiny",
}


EMBED_MODEL = {
    "bgem3": "llm/embedder/bge-m3",
    "minilml6v2": "sentence-transformers/all-MiniLM-L6-v2",
    "minilml12v2": "sentence-transformers/all-MiniLM-L12-v2",
    "hitsnomed": "llm/embedder/HiT-MiniLM-L12-SnomedCT"
}


# quantize the LLM if needed
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


# LLM pipeline module, used for initializing the LLM model for extractors, parsers, and generators
def init_llm_pipeline(llm_model_name, quantization_config=quantization_config):

    if llm_model_name.find("gpt") != -1:
        pipe = OpenAI()
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, 
                                                     device_map="cuda:0",
                                                     torch_dtype="auto", 
                                                     quantization_config=quantization_config,
                                                     trust_remote_code=True,)
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(f"LLM pipeline built: {llm_model_name}")
    return pipe


# LLM entity extractor module, a seperate module used for extracting entities from the text
def llm_entity_extractor(text, pipe, using_extractor):
    prompt_tmpl = f"Extract all entities exhaustively from the following text: {text}. \n ONLY respond with the ENTITIES without any reasoning. \n Entities: []"

    messages = [
                # {
                #     "role": "system",
                #     "content": "You are a friendly chatbot who always responds in the style of a pirate",
                # },
                {"role": "user", "content": prompt_tmpl.format(text=text)},
            ]
    
    if using_extractor.find("gpt") != -1:
        completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

        # print(completion.choices[0].message.content)
        entities = completion.choices[0].message.content
    
    else:
        messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        entities = pipe(messages, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=100, top_p=0.95)
    
    logging.info(f"Entities extracted: {entities}")
    return entities


# LLM Parser module, used for parsing the output of the RAG or LLM extractor
def llm_parser(text, pipe, using_parser="nuparser", target_results="Concepts"):

    prompt_tmpl = f"Here is the result of RAG: {text}. \n Parse the result by extracting all {target_results} and removing all repetitions. \n {target_results}: []"

    messages = [ {"role": "user", 
                  "content": prompt_tmpl.format(text=text)} 
                  ]
    
    if using_parser.find("gpt") != -1:
        completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

        results = completion.choices[0].message.content

    elif using_parser.find("nuparser") != -1:
        parser_template = """{
            "%s": []
        }""" % target_results
        parser_template = json.dumps(json.loads(parser_template), indent=4)
        prompt = f"""<|input|>\n### Template:\n{parser_template}\n### Text:\n{text}\n\n<|output|>"""
        
        with torch.no_grad():
            messages = pipe.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=10000).to(pipe.model.device)

            pred_ids = pipe.model.generate(**messages, max_new_tokens=4000)
            results = pipe.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            results = results[0].split("<|output|>")[1]

    else:
        messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        results = pipe(messages, 
                    max_new_tokens=2048, 
                    do_sample=True, 
                    temperature=0.7, 
                    repetition_penalty=1.5,
                    top_k=10, 
                    top_p=0.95)
    
    # print(f"{target_results} extracted: {results}")
    return results


# LLM generator module, used for generating the response from the LLM model
def llm_generator(prompt_tmpl, pipe, using_generator):

    messages = [
                # {
                #     "role": "system",
                #     "content": "You are a friendly chatbot who always responds in the style of a pirate",
                # },
                {"role": "user", "content": prompt_tmpl},
            ]
    
    if using_generator.find("gpt") != -1:
        completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )

        # print(completion.choices[0].message.content)
        pairs = completion.choices[0].message.content
    
    else:
        messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pairs = pipe(messages, max_new_tokens=2048, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    logging.info(f"pairs extracted: {pairs}")
    return pairs


# LLM service context module in the RAG pipeline, used for initializing the LLM model and the embedding model 
def init_llm_service_context(llm_model_name="HuggingFaceH4/zephyr-7b-alpha", 
                             tokenizer_name="HuggingFaceH4/zephyr-7b-alpha", 
                             embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
                             quantization_config=quantization_config, 
                             context_window=32768,
                             max_new_tokens=256,
                             token=None, 
                             output_parser=None,):
    if "gpt" in llm_model_name:
        llm = OpenAI(model=llm_model_name, api_key=OPENAI_API_KEY, temperature=0.7)
        # embed_model = OpenAIEmbedding(model_name="text-embeddings-ada-002", api_key=os.environ["OPENAI_API_KEY"])
        embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
    elif "deepseek" in llm_model_name:
        logging.info("Using Deepseek API")
        llm = DeepSeek(model=llm_model_name, api_key=DEEPSEEK_API_KEY, temperature=0.7)
        embed_model = HuggingFaceEmbeddings(model_name=embed_model_name)
    else:
        llm = HuggingFaceLLM(
            model_name=llm_model_name, # epfl-llm/meditron-7b
            tokenizer_name=tokenizer_name, # epfl-llm/meditron-7b
            # query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            model_kwargs={"quantization_config": quantization_config, "trust_remote_code": True},
            # tokenizer_kwargs={"trust_remote_code": True,},
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True},
            # messages_to_prompt=messages_to_prompt,
            device_map="cuda:0",
            output_parser=output_parser,
        )

        embed_model = HuggingFaceEmbeddings(model_name=embed_model_name) 
    Settings.llm = llm
    Settings.embed_model = embed_model
    logging.info(f"LLM loaded: {Settings.llm.model_name}" if not isinstance(Settings.llm, OpenAI) else f"LLM loaded: {Settings.llm.model}")
    logging.info(f"embed_model loaded: {Settings.embed_model.model_name}")
    logging.info("Settings loaded.")

    return llm #, service_context


# KG storage context module in the RAG pipeline, used for initializing the KG index
def init_kg_storage_context(llm, storage_dir="data/index/ade_full_graph_doc"):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    logging.info(f"KG storage: {storage_dir}")
    logging.info("KG Storage context loaded")

    kg_index = load_index_from_storage(
        storage_context=storage_context,
        # service_context=service_context,
        llm=llm,
    )
    logging.info("KG index loaded")

    return kg_index


# Vector storage context module in the RAG pipeline, used for initializing the vector storage
def init_vector_storage_context(storage_dir="data/index/ade_full_vector"):
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    logging.info(f"Vector storage: {storage_dir}")
    logging.info("Vector storage context loaded")

    return storage_context   


# Aggregated RAG pipeline module, used for initializing the RAG pipeline
def init_rag_pipeline(kg_index,
                      include_text=False, 
                      similarity_top_k=3, 
                      graph_store_query_depth=2, 
                      retriever_mode="hybrid",
                      verbose=False
                      ):

    kg_query_engine = kg_index.as_query_engine(
        include_text=include_text,
        # retriever_mode="hybrid",
        retriever_mode=retriever_mode,
        embedding_mode="hybrid",
        response_mode="compact",
        similarity_top_k=similarity_top_k,
        graph_store_query_depth=graph_store_query_depth,
        explore_global_knowledge=True,
        verbose=verbose,
    )

    logging.info("RAG pipeline created")
    return kg_query_engine

