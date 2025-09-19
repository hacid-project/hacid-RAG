import os, json, ast
import torch
import sys, re
import logging
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import openai
from openai import OpenAI
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") if os.getenv("DEEPSEEK_API_KEY") else ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else ""
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") if os.getenv("NVIDIA_API_KEY") else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from llama_index.core import KnowledgeGraphIndex, Settings, ServiceContext, SimpleDirectoryReader, download_loader, load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
from IPython.display import Markdown, display
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core.indices.knowledge_graph.retrievers import KGTableRetriever
from llama_index.core.prompts.default_prompts import DEFAULT_KEYWORD_EXTRACT_TEMPLATE
from llama_index.core.indices.keyword_table.utils import extract_keywords_given_response
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from utils.deepseek_llm import DeepSeek
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index.core.embeddings.openai import OpenAIEmbedding

LLM = {
    "commandr": "../llm/c4ai-command-r-v01-4bit",
    "zephyralpha": "HuggingFaceH4/zephyr-7b-alpha",
    "zephyrbeta": "HuggingFaceH4/zephyr-7b-beta",
    "mistralsmall": "../llm/Mistral-Small-Instruct-2409",
    "mistralsmall7B": "../llm/Mistral-7B-Instruct-v0.3",
    "llama3.18b": "../llm/Llama-3.1-8B-Instruct",
    "gpt-4o-mini": "gpt-4o-mini",
    "ds-r1-llama": "../llm/DeepSeek-R1-Distill-Llama-8B",
    "ds-r1-qwen": "../llm/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-chat": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
    # "nuparser-tiny": "../llm/NuExtract-1.5-tiny",
    # "nuparser": "../llm/NuExtract-1.5",
    "mistral-ft-multitask": "../../LLM-Medical-Finetuning/finetuned_models/Mistral-Small-Instruct-2409-multitask_240703/_merged",
}


EMBED_MODEL = {
    "bgem3": "../llm/embedder/bge-m3",
    "minilml6v2": "sentence-transformers/all-MiniLM-L6-v2",
    "minilml12v2": "sentence-transformers/all-MiniLM-L12-v2",
    "medcptQuery": "../llm/embedder/ncbi_MedCPT-Query-Encoder",
    "hitsnomed": "../llm/embedder/HiT-MiniLM-L12-SnomedCT"
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
        tokenizer = AutoTokenizer.from_pretrained(LLM["mistralsmall"])
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, 
                                                     device_map="auto",
                                                     torch_dtype="auto", 
                                                     quantization_config=quantization_config,
                                                     trust_remote_code=True,)
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # pipe = pipeline("text-generation", model=model)

    logging.info(f"LLM pipeline built: {llm_model_name}")
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
def llm_generation(prompt_tmpl, pipe, using_generator):

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
        results = completion.choices[0].message.content
    
    else:
        messages = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        results = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.05, top_k=50, top_p=0.95)
    
    # logging.info(f"Extractions: {results}")
    return results


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
            generate_kwargs={"temperature": 0.65, "top_k": 50, "top_p": 0.95, "do_sample": True},
            # messages_to_prompt=messages_to_prompt,
            # device_map="auto",
            output_parser=output_parser,
        )

        # llm = HuggingFaceInferenceAPI(
        #     model_name="mistralai/Mistral-Small-Instruct-2409",
        #     temperature=0.7,
        #     context_window=context_window,
        #     max_tokens=max_new_tokens,
        #     token="",  # Optional
        # )

        embed_model = HuggingFaceEmbeddings(model_name=embed_model_name) 
    # service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512, embed_model=embed_model)
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


class CustomKGTableRetriever(KGTableRetriever):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ollama embedding model configuration
        self.embedding_model = 'nomic-embed-text:137m-v1.5-fp16'
        logging.info(f"CustomKGTableRetriever is using the embedding model: {self.embedding_model}")
        
        # The predefined list of relation predicates
        self.relation_predicates = [
            'after', 'associated aetiologic finding', 'associated etiologic finding', 'associated finding',
            'associated morphology', 'associated with', 'causative agent', 'course', 'direct morphology',
            'direct substance', 'due to', 'finding site', 'has active ingredient', 'has definitional manifestation',
            'has realization', 'interprets', 'is modification of', 'measures', 'part of', 'pathological process',
            'temporally follows', 'temporally related to', 'type', 'using'
        ]

        # Parameter configuration
        self.max_per_relation = 5
        self.similarity_threshold = 0.45
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3
        self.keywords = []

        # Stopwords list
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall'
        }
        
        # Important relations for weighting
        self.important_relations = [
            'associated morphology', 'due to', 'causative agent', 'pathological process',
            'finding site', 'direct morphology', 'has definitional manifestation'
        ]

    def get_ollama_embedding(self, text):
        """
        get embedding from Ollama, handle formatting and errors
        
        Args:
            test: input text (ensure it's a string)
            
        Returns:
            2D numpy array of the embedding
        """
        try:
            # ensure input is a string
            if isinstance(text, list):
                text = ' '.join(text)
            elif not isinstance(text, str):
                text = str(text)

            # call ollama embedding API
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = np.array(response['embedding'])

            # ensure it's 2D array format (1, n_features)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
                
            return embedding
            
        except Exception as e:
            print(f"[DEBUG] Ollama embedding computation failed: {e}")
            # return zero vector as fallback
            return np.zeros((1, 768))  # assuming 768-dim embeddings

    def parse_triple_string(self, triple_str):
        """safely convert string-formatted triple to tuple"""
        try:
            parsed = ast.literal_eval(triple_str)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                return parsed[0], parsed[1], parsed[2]
            else:
                return None
        except (ValueError, SyntaxError):
            return None

    def extract_keywords(self, text, nodewithscores=None):
        """extract keywords from text"""
        # words = re.findall(r'\b\w+\b', text.lower())
        # keywords = [word for word in words if len(word) > 2 and word not in self.stopwords]

        keywords = []
        if nodewithscores:
            for i, nws in enumerate(nodewithscores):
                node = nws.node
                kg_rel_map = node.metadata.get('kg_rel_map', [])

            print(f"[DEBUG] Node {i} contains {len(kg_rel_map)} keywords")
            
            for j, keyword in enumerate(kg_rel_map):
                if isinstance(keyword, str) and keyword not in self.stopwords:
                    keywords.append(keyword)
        else:
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [word for word in words if len(word) > 2 and word not in self.stopwords]

        return keywords

    def compute_semantic_similarity(self, query_text, triple):
        """
        compute semantic similarity using Ollama embeddings
        
        Args:
            query_text: the input query text
            triple: (subject, relation, object) triple

        Returns:
            semantic similarity score (0-1)
        """
        subj, rel, obj = triple

        # construct the representation of the triple
        triple_text = f"{subj} {rel} {obj}"
        
        try:
            # call Ollama to get embeddings
            query_embedding = self.get_ollama_embedding(query_text)
            triple_embedding = self.get_ollama_embedding(triple_text)

            # compute cosine similarity
            similarity = cosine_similarity(query_embedding, triple_embedding)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            print(f"[DEBUG] Semantic similarity computation failed: {e}")
            return 0.0

    def compute_enhanced_semantic_similarity(self, query_text, triple):
        """
        compute enhanced semantic similarity by combining triple-level and entity-level similarities

        Args:
            query_text: the input query text
            triple: (subject, relation, object) triple

        Returns:
            enhanced semantic similarity score (0-1)
        """
        subj, rel, obj = triple
        
        try:
            # 1. Overall triple similarity
            overall_sim = self.compute_semantic_similarity(query_text, triple)
            
            # 2. Entity-level similarity
            query_embedding = self.get_ollama_embedding(query_text)
            
            # 分别计算subject和object的相似度
            subj_embedding = self.get_ollama_embedding(subj)
            obj_embedding = self.get_ollama_embedding(obj)
            
            subj_sim = cosine_similarity(query_embedding, subj_embedding)[0][0]
            obj_sim = cosine_similarity(query_embedding, obj_embedding)[0][0]
            
            max_entity_sim = max(subj_sim, obj_sim)
            
            # 3. Relation weighting
            relation_weight = 1.2 if rel in self.important_relations else 1.0
            
            # 60% overall + 40% entity-level, then apply relation weight
            final_score = (0.6 * overall_sim + 0.4 * max_entity_sim) * relation_weight
            
            return min(final_score, 1.0)
            
        except Exception as e:
            print(f"[DEBUG] Enhanced semantic similarity computation failed: {e}")
            return 0.0

    def compute_keyword_similarity(self, query_text, triple):
        """compute keyword similarity based on keyword matching"""
        subj, rel, obj = triple
        
        # query_keywords = set(self.extract_keywords(query_text))
        triple_text = f"{subj} {obj}"
        triple_keywords = set(self.extract_keywords(triple_text))
        
        if len(self.keywords) == 0:
            return 0.0

        common_keywords = self.keywords.intersection(triple_keywords)
        union_keywords = self.keywords.union(triple_keywords)

        jaccard_sim = len(common_keywords) / len(union_keywords) if len(union_keywords) > 0 else 0.0
        coverage = len(common_keywords) / len(self.keywords)

        final_score = 0.6 * jaccard_sim + 0.4 * coverage
        return final_score

    def compute_hybrid_similarity(self, query_text, triple):
        """compute hybrid similarity (semantic + keyword)"""
        semantic_score = self.compute_enhanced_semantic_similarity(query_text, triple)
        keyword_score = self.compute_keyword_similarity(query_text, triple)
        
        hybrid_score = self.semantic_weight * semantic_score + self.keyword_weight * keyword_score
        return hybrid_score

    def extract_triples_with_hybrid_scores(self, nodewithscores, query_text):
        """extract triples and compute hybrid scores"""
        print(f"[DEBUG] Starting hybrid scoring computation (Ollama)...")

        self.keywords = self.extract_keywords(query_text, nodewithscores)
        print(f"[DEBUG] Query keywords: {self.keywords[:10]}")
        self.keywords = set(self.keywords)  # remove duplicates

        triples_with_scores = []

        for i, nws in enumerate(nodewithscores):
            node = nws.node
            kg_rel_texts = node.metadata.get('kg_rel_texts', [])

            print(f"[DEBUG] Node {i} contains {len(kg_rel_texts)} triples")

            for j, triple_str in enumerate(kg_rel_texts):
                parsed_triple = self.parse_triple_string(triple_str)

                if parsed_triple:
                    # compute hybrid similarity
                    hybrid_score = self.compute_hybrid_similarity(query_text, parsed_triple)

                    subj, rel, obj = parsed_triple
                    triples_with_scores.append((subj, rel, obj, hybrid_score))

                    # show detailed scores for first 5 triples of each node
                    if j < 5:
                        semantic_score = self.compute_enhanced_semantic_similarity(query_text, parsed_triple)
                        keyword_score = self.compute_keyword_similarity(query_text, parsed_triple)
                        print(f"[DEBUG] Triple {j}: {rel}")
                        print(f"Semantic score: {semantic_score:.3f}, Keyword score: {keyword_score:.3f}, Hybrid score: {hybrid_score:.3f}")

        # sort by hybrid score
        triples_with_scores.sort(key=lambda x: x[3], reverse=True)

        if triples_with_scores:
            print(f"[DEBUG] Hybrid score range: {triples_with_scores[-1][3]:.3f} - {triples_with_scores[0][3]:.3f}")

        return triples_with_scores

    def post_process_results(self, retrieved_triples: List[Tuple[str, str, str, float]]) -> List[Tuple[str, str, str]]:
        """post-process retrieved triples to enforce relation balance and deduplication"""
        print(f"[DEBUG] Post-processing started with {len(retrieved_triples)} triples")

        # similarity filtering
        filtered = [t for t in retrieved_triples if t[3] >= self.similarity_threshold]
        print(f"[DEBUG] After similarity filtering: {len(filtered)} triples")

        if len(filtered) == 0:
            print("[DEBUG] Warning: No triples after similarity filtering! Consider lowering similarity_threshold")
            return []

        # sort by score
        filtered.sort(key=lambda x: x[3], reverse=True)

        # relation type statistics
        relation_stats = defaultdict(int)
        for triple in filtered:
            relation_stats[triple[1]] += 1
        print(f"[DEBUG] Relation type distribution: {dict(relation_stats)}")

        # relation type balancing and deduplication
        relation_count = defaultdict(int)
        balanced = []
        seen = set()
        excluded_relations = set()

        for triple in filtered:
            subj, rel, obj, score = triple

            if rel not in self.relation_predicates:
                excluded_relations.add(rel)
                continue

            if relation_count[rel] < self.max_per_relation:
                triple_key = (subj, rel, obj)

                if triple_key not in seen:
                    balanced.append(triple_key)
                    relation_count[rel] += 1
                    seen.add(triple_key)

        print(f"[DEBUG] Excluded relation types: {excluded_relations}")
        print(f"[DEBUG] Final result count: {len(balanced)}")
        print(f"[DEBUG] Final relation distribution: {dict(relation_count)}")

        return balanced

    def retrieve(self, query_text, *args, **kwargs):
        """the main retrieval method"""
        print("[DEBUG] Starting hybrid scoring retrieval (Ollama version)...")

        nodewithscores = super().retrieve(query_text, *args, **kwargs)
        print(f"[DEBUG] Parent retriever returned {len(nodewithscores)} NodeWithScore objects")

        if len(nodewithscores) == 0:
            return []

        triples_with_scores = self.extract_triples_with_hybrid_scores(nodewithscores, query_text)

        if len(triples_with_scores) == 0:
            return []

        processed_triples = self.post_process_results(triples_with_scores)

        print(f"[DEBUG] Hybrid scoring retrieval done, returning {len(processed_triples)} triples")
        return processed_triples


# Ensure Ollama is installed
try:
    import ollama
except ImportError:
    raise ImportError("The 'ollama' package is required but not installed. Please install it using 'pip install ollama'.")

def init_retriever(kg_index,
                   include_text=False,
                   similarity_top_k=10,
                   graph_store_query_depth=2,
                   retriever_mode="hybrid",
                   verbose=False,
                   max_knowledge_sequence=100,
                   custom_retriever=False, # the default is False to use the original KGTableRetriever, if True, use the CustomKGTableRetriever
                   ):
        
    if custom_retriever:
        retriever = CustomKGTableRetriever(
            index=kg_index,
            similarity_top_k=similarity_top_k,
            graph_store_query_depth=graph_store_query_depth,
            verbose=verbose,
            include_text=include_text,
            retriever_mode=retriever_mode,
            max_knowledge_sequence=max_knowledge_sequence,
        )
    else:
        retriever = kg_index.as_retriever(
            similarity_top_k=similarity_top_k,
            graph_store_query_depth=graph_store_query_depth,
            verbose=verbose,
            retriever_mode=retriever_mode,
        )

    print(f"Retriever created, retriever: {type(retriever)}, retriever_mode: {retriever_mode}")
    return retriever


# Aggregated RAG pipeline module, used for initializing the RAG pipeline
def init_rag_pipeline(kg_index,
                      include_text=False, 
                      similarity_top_k=3, 
                      graph_store_query_depth=2, 
                      retriever_mode="hybrid",
                      verbose=False
                      ):

    # retriever = init_retriever(
    #     kg_index = kg_index,
    #     similarity_top_k=similarity_top_k,
    #     graph_store_query_depth=graph_store_query_depth,
    #     verbose=verbose,
    #     include_text=include_text,
    #     retriever_mode=retriever_mode,
    # )

    # kg_query_engine = RetrieverQueryEngine.from_args(
    #     retriever=retriever,
    #     response_mode="compact",
    #     verbose=verbose,
    # )

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

    logging.info(f"RAG pipeline created, RAG pipeline: {type(kg_query_engine)}")
    return kg_query_engine

