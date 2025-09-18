# Description: This script is used to extract relations from the RAG model using various prompt templates.

import datetime
from datetime import datetime
from tqdm import tqdm
from rag_prompt_template import *
from rag_util import *
from rag_moduler import *
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--task', type=str, required=True, choices=['concept', 'pair', 'triple', 'description', 'diagnosis'], help='The task to perform')
    parser.add_argument('--using_llm', type=str, required=True, help='The LLM model to use')
    parser.add_argument('--using_embed', type=str, required=True, help='The embedding model to use')
    parser.add_argument('--eval_dataset', type=str, required=True, choices=['BC5CDR', 'NCBI', 'MIMICIV', 'MIMICIV-chunk', 'pubmed', 'humandx'], help='The evaluation dataset to use')
    parser.add_argument('--topk', type=int, required=True, help='The top k to retrieve')
    parser.add_argument('--depth', type=int, required=True, help='The depth to query')
    parser.add_argument('--retrieve_mode', type=str, required=True, help='The retrieve mode to use')
    parser.add_argument('--using_extractor', type=str, required=False, help='The helper extractor to use')
    parser.add_argument('--using_parser', type=str, required=False, help='The parser to use')
    parser.add_argument('--note', type=str, required=False, help='Any additional note, e.g. with or without human diagnosis')
    
    args = parser.parse_args()
    return args


def entity_extraction(text, pipe_extractor, using_extractor):
    if using_extractor == "mistralsmall":
        entities = llm_entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)[0]["generated_text"].split("Entities: ")[1].replace("[][/INST]", "").strip()
    elif using_extractor == "solar":
        entities = llm_entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)
        if "assistant\Entities:" in entities:
            entities = entities[0]["generated_text"].split("assistant\Entities:")[1].strip()
        else:
            entities = entities[0]["generated_text"].split("Entities:")[1].strip()
    elif using_extractor == "gpt-4o-mini":
        entities = llm_entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)
    return entities

# Deprecated
'''
def query_and_generate_rel(test_id, query_engine, input_text_dir, cases=427):
    logging.info(f"Query Engine: {query_engine}")

    fmt_prompt = prompt_tmpl.format(text="Test")
    logging.info(f"Prompt: {fmt_prompt}")

    with open(input_text_dir) as f:
    # with open("data/chemprot/test.source") as f:
        results = []
        if input_text_dir.endswith(".json"):
            sentences = [sample["abstract"] for sample in json.load(f)]
        else:
            sentences = f.readlines()
        logging.info(f"Experiment ID: {test_id}")
        logging.info(f"Number of sentences: {len(sentences)}; Number of cases for test: {cases}")
        for sentence_id, text in tqdm(enumerate(sentences[:cases])):
            try:
                logging.info(f"Processing sentence {sentence_id} / {len(sentences)}")

                if using_extractor is not None:
                    entities = entity_extraction(text, pipe_extractor, using_extractor)

                    fmt_prompt = prompt_tmpl.format(
                        text=text,
                        entities=entities
                    )
                else:
                    fmt_prompt = prompt_tmpl.format(
                        text=text
                    )
                # print(fmt_prompt)
                response = query_engine.query(fmt_prompt)
                display(Markdown(f"<b>{response}</b>"))
                while "Concepts: " not in str(response):
                    logging.info(f"Response did not contain 'Concepts: '. Retrying query...")
                    response = query_engine.query(fmt_prompt)
                results.append(extract_triple(str(response), split_str1="Concepts: ") + "\n")
                print(f"======= Sentence {sentence_id} processed.. ========")
            except Exception as e:
                logging.error(f"Error processing sentence {sentence_id} / {len(sentences)}")
                logging.error(f"Error: {e}")
                results.append("Error\n")
                continue

    with open(f"results/{test_id}", 'w') as f:
        f.writelines(results)
    
    logging.info(f"Results saved to results/rel.hyps_{test_id}")
'''

def rag_init():
    logging_setup(log_file=f"logs/{PARAMETERS['test_id']}.log", log_level=logging.INFO)
    logging.info(f"PARAMETERS: {PARAMETERS}")
    # logging.info(f"Prompt: {using_prompt}")

    llm = init_llm_service_context(llm_model_name=PARAMETERS["llm_model_name"], 
                                    tokenizer_name=PARAMETERS["tokenizer_name"], 
                                    embed_model_name=PARAMETERS["embed_model_name"],
                                    context_window=PARAMETERS["context_window"],
                                    max_new_tokens=PARAMETERS["max_new_tokens"],
                                    # quantization_config=None,
                                )
    
    pipe_extractor = init_llm_pipeline(LLM[using_extractor]) if using_extractor is not None else None
    pipe_parser = init_llm_pipeline(LLM[using_parser], quantization_config=None) if using_parser is not None else None

    kg_index = init_kg_storage_context(llm, storage_dir=PARAMETERS["storage_dir"])
    # kg_index = init_kg_storage_context(llm, storage_dir="index/snomed_dataset_nodoc_commandr_minilml6v2")

    query_engine = init_rag_pipeline(kg_index, 
                                     similarity_top_k=PARAMETERS["similarity_top_k"], 
                                     graph_store_query_depth=PARAMETERS["graph_store_query_depth"], 
                                     include_text=False, 
                                     retriever_mode=PARAMETERS["retriever_mode"],
                                     verbose=PARAMETERS["verbose"])
    
    return query_engine, pipe_extractor, pipe_parser

# start query and extract information
def rag_start(test_id, 
              input_text_dir, 
              query_engine,
              prompt_tmpl, 
              target_results, 
              pipe_extractor=None, 
              pipe_parser=None, 
              cases=0
              ):
    logging.info(f"Query Engine: {query_engine}")
    logging.info(f"Extractor: {pipe_extractor.model}") if pipe_extractor is not None else logging.info("Extractor: None")
    logging.info(f"Parser: {pipe_parser.model}") if pipe_parser is not None else logging.info("Parser: None")

    fmt_prompt = prompt_tmpl.format(text="Test")
    logging.info(f"Prompt: {fmt_prompt}")

    with open(input_text_dir) as f:
        results = []
        if input_text_dir.endswith(".json"):
            sentences = [sample["abstract"] for sample in json.load(f)]
        else:
            sentences = f.readlines()
        logging.info(f"Experiment ID: {test_id}")
        
        if cases == 0:
            cases = len(sentences)
            
        logging.info(f"Number of sentences: {len(sentences)}; Number of cases for test: {cases}")
        for sentence_id, text in tqdm(enumerate(sentences[:cases])):
            try:
                logging.info(f"Processing sentence {sentence_id} / {len(sentences)}")

                if pipe_extractor is not None:
                    entities = entity_extraction(text, pipe_extractor, using_extractor)

                    fmt_prompt = prompt_tmpl.format(
                        text=text,
                        entities=entities
                    )
                else:
                    fmt_prompt = prompt_tmpl.format(
                        text=text
                    )
                # print(fmt_prompt)
                response = query_engine.query(fmt_prompt)
                # print(f"response: {response}")
                parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                structured_result = clean_structural_list(parsed_response, target_results)
                retry_count = 0
                while len(structured_result) == 0:
                    if retry_count > 3: # if the response is still empty after 3 retries, break
                        structured_result = []
                        break
                    logging.info(f"Response did not contain any {target_results}, Retrying query... (Retry {retry_count})")
                    response = query_engine.query(fmt_prompt)
                    parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results).strip()
                    structured_result = clean_structural_list(parsed_response, target_results)
                    retry_count += 1
                results.append(str(structured_result) + "\n")
                print(f"======= Sentence {sentence_id} processed.. ========")
            except Exception as e:
                logging.error(f"Error processing sentence {sentence_id} / {len(sentences)}")
                logging.error(f"Error: {e}")
                results.append("Error\n")
                continue

    with open(f"results/{test_id}", 'w') as f:
        f.writelines(results)
    
    logging.info(f"Results saved to results/rel.hyps_{test_id}")


def rag_start_diagnosis(test_id, 
                        input_text_dir, 
                        query_engine,
                        prompt_tmpl, 
                        target_results, 
                        pipe_extractor=None, 
                        pipe_parser=None, 
                        cases=0
                        ):
    logging.info(f"Query Engine: {query_engine}")
    logging.info(f"Extractor: {pipe_extractor.model}") if pipe_extractor is not None else logging.info("Extractor: None")
    logging.info(f"Parser: {pipe_parser.model}") if pipe_parser is not None else logging.info("Parser: None")

    fmt_prompt = prompt_tmpl.format(text="Test")
    logging.info(f"Prompt: {fmt_prompt}")

    with open(input_text_dir) as f:
        results = []
        all_cases = json.load(f)
        logging.info(f"Experiment ID: {test_id}")
        
        if cases == 0:
            cases = len(all_cases)
            
        logging.info(f"Number of Case: {len(all_cases)}; Number of cases for test: {cases}")
        for case_id, case_dict in tqdm(enumerate(all_cases[:cases])):
            try:
                logging.info(f"Processing sentence {case_id} / {len(all_cases)}")

                if note.find("HDX") != -1: # if the note contains HDX, use the HDX prompt
                    all_human_dx = prepare_hdx(case_dict["human_dx"])
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"],
                        all_human_dx=all_human_dx
                    )
                
                else:
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"]
                    )
                    
                # print(fmt_prompt)
                response = query_engine.query(fmt_prompt)
                if using_llm.find("ds") != -1:
                    response = response.response.split("</think>")[1]
                # print(f"response: {response}")
                parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                structured_result = clean_structural_list(parsed_response, target_results)
                while len(structured_result) == 0:
                    logging.info(f"Response did not contain any {target_results}, Retrying query...")
                    response = query_engine.query(fmt_prompt)
                    if using_llm.find("ds") != -1:
                        response = response.response.split("</think>")[1]
                    parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                    structured_result = clean_structural_list(parsed_response, target_results)
                results.append(str(structured_result) + "\n")
                print(f"======= Case {case_id} processed.. ========")
            except Exception as e:
                logging.error(f"Error processing Case {case_id} / {len(all_cases)}")
                logging.error(f"Error: {e}")
                results.append("Error\n")
                continue

    with open(f"results_diagnosis/{test_id}", 'w') as f:
        f.writelines(results)
    
    logging.info(f"Results saved to results_diagnosis/diagnosis_{test_id}")


if __name__ == '__main__':

    args = parse_arguments()
    task = args.task
    using_llm = args.using_llm if args.using_llm else "mistralsmall"
    using_embed = args.using_embed if args.using_embed else "hitsnomed"
    eval_dataset = args.eval_dataset if args.eval_dataset else "BC5CDR"
    similarity_top_k = args.topk if args.topk else 30
    graph_store_query_depth = args.depth if args.depth else 5
    retrieve_mode = args.retrieve_mode if args.retrieve_mode else "hybrid"
    using_extractor = args.using_extractor
    using_parser = args.using_parser if args.using_parser else "nuparser"
    note = args.note
    quantization = "Q"
    date = datetime.now().strftime("%Y%m%d_%H%M%S")

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM["mistralsmall"],
        "embed_model_name": EMBED_MODEL[using_embed],
        "storage_dir": f"index/snomed_all_dataset_nodoc_{using_embed}",
        # "storage_dir": f"index/snomed_dataset_nodoc_commandr_hitsnomed", # this is a partial KG indices for testing
        "input_text_dir": EVAL_DATA[eval_dataset],
        # "input_text_dir": "" # uncomment this line to use the custom input text
        "context_window": 32768,
        "max_new_tokens": 512,
        "case_num":3833,
        "verbose": True,
        "quantization": None,
        "similarity_top_k": similarity_top_k,
        "graph_store_query_depth": graph_store_query_depth,
        "retriever_mode": retrieve_mode,
        "test_id": f"{date}_{task}_{eval_dataset}_{using_llm}_{using_embed}_topk{similarity_top_k}_depth{graph_store_query_depth}_{retrieve_mode}_extractor_{using_extractor}_quantization_{quantization}_note_{note}"
    }

    query_engine, pipe_extractor, pipe_parser = rag_init()

    if task == "diagnosis":
        if note.find("HDX") != -1: # if the note contains HDX, use the HDX prompt
            using_prompt = Humandx_case_diagnosis_prompt_HDX_shorthand
        else:
            using_prompt = Humandx_case_diagnosis_prompt_shorthand

        prompt_tmpl = PromptTemplate(
            using_prompt,
            template_var_mappings=prompt_var_mappings
        )

        for ep in range(1, 2):
            rag_start_diagnosis(test_id=PARAMETERS["test_id"] + f"_ep{ep}",
                                input_text_dir=PARAMETERS["input_text_dir"],
                                prompt_tmpl=prompt_tmpl,
                                target_results="Diagnosis",
                                query_engine=query_engine,
                                pipe_extractor=pipe_extractor,
                                pipe_parser=pipe_parser,
                                cases=PARAMETERS["case_num"],
                                )
    else:
        if using_extractor is None: # w/o extractor mode
            using_prompt = finetuned_MIMICIV_entity_extraction_prompt
        else: # with extractor mode
            using_prompt = IoU_MIMICIV_entity_extraction_prompt_with_entities

        prompt_tmpl = PromptTemplate(
            using_prompt,
            template_var_mappings=prompt_var_mappings
        )

        rag_start(test_id=PARAMETERS["test_id"],
                    input_text_dir=PARAMETERS["input_text_dir"],
                    prompt_tmpl=prompt_tmpl,
                    target_results="Concepts",
                    query_engine=query_engine,
                    pipe_extractor=pipe_extractor,
                    pipe_parser=pipe_parser,
                    cases=PARAMETERS["case_num"],
                    )
    
    logging.info("Done!")