# Description: This script is used to extract relations from the RAG model using various prompt templates.

from tqdm import tqdm
from src.rag_prompt_template import *
from rag_util import *
from src.rag_moduler import *
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--task', type=str, required=True, choices=['concept', 'pair', 'triple', 'description', 'diagnosis'], help='The task to perform')
    parser.add_argument('--using_llm', type=str, required=True, help='The LLM model to use')
    parser.add_argument('--using_embed', type=str, required=True, help='The embedding model to use')
    parser.add_argument('--eval_dataset', type=str, required=True, choices=['BC5CDR', 'NCBI', 'MIMICIV', 'pubmed', 'humandx'], help='The evaluation dataset to use')
    parser.add_argument('--topk', type=int, required=True, help='The top k to retrieve')
    parser.add_argument('--depth', type=int, required=True, help='The depth to query')
    parser.add_argument('--retrieve_mode', type=str, required=True, help='The retrieve mode to use')
    parser.add_argument('--using_extractor', type=str, required=False, help='The helper extractor to use')
    parser.add_argument('--using_parser', type=str, required=False, help='The parser to use')
    parser.add_argument('--note', type=str, required=False, help='Any additional note, e.g. with or without human diagnosis')
    
    args = parser.parse_args()
    return args


def gen_diagnosis(fmt_prompt, pipe_generator, using_llm, target_results):

    response = llm_generation(fmt_prompt, pipe_generator, using_llm)

    if using_llm.find("gpt") != -1:
        parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
    elif using_llm.find("mistral") != -1:
        parsed_response = llm_parser(response[0]["generated_text"].split("[/INST]")[1], pipe_parser, "nuparser", target_results)
    elif using_llm.find("commandr") != -1:
        parsed_response = llm_parser(response[0]["generated_text"].split("<|CHATBOT_TOKEN|>")[1], pipe_parser, "nuparser", target_results)
    elif using_llm.find("solar") != -1:
        parsed_response = llm_parser(response[0]["generated_text"].split("<|im_start|>assistant")[1], pipe_parser, "nuparser", target_results)
    elif using_llm.find("ds") != -1:
        parsed_response = llm_parser(response[0]["generated_text"].split("</think>")[1], pipe_parser, "nuparser", target_results)
    structured_result = clean_structural_list(parsed_response, target_results)
    
    return structured_result


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
    
    pipe_generator = init_llm_pipeline(LLM[using_llm])
    pipe_parser = init_llm_pipeline(LLM[using_parser], quantization_config=None) if using_parser is not None else None

    kg_index = init_kg_storage_context(llm, storage_dir=PARAMETERS["storage_dir"])
    # kg_index = init_kg_storage_context(llm, storage_dir="index/snomed_dataset_nodoc_commandr_minilml6v2")

    query_engine = init_rag_pipeline(kg_index, 
                                     similarity_top_k=PARAMETERS["similarity_top_k"], 
                                     graph_store_query_depth=PARAMETERS["graph_store_query_depth"], 
                                     include_text=False, 
                                     retriever_mode=PARAMETERS["retriever_mode"],
                                     verbose=PARAMETERS["verbose"])
    
    retriever = init_retriever(kg_index=kg_index,
                                    similarity_top_k=PARAMETERS["similarity_top_k"],
                                    graph_store_query_depth=PARAMETERS["graph_store_query_depth"],
                                    verbose=True,
                                    retriever_mode=PARAMETERS["retriever_mode"],
                                    )
    
    return query_engine, pipe_generator, pipe_parser, retriever

# a series connecting the RAG model with the LLM
def llm_rag_start_diagnosis(test_id, 
                        input_text_dir, 
                        target_results,
                        prompt_tmpl=None,
                        query_engine=None, 
                        pipe_generator=None, 
                        pipe_parser=None, 
                        retriever=None,
                        cases=0
                        ):
    logging.info(f"Query Engine: {query_engine}")
    logging.info(f"LLM Generator: {pipe_generator.model}")
    logging.info(f"Parser: {pipe_parser.model}") if pipe_parser is not None else logging.info("Parser: None")

    if prompt_tmpl is not None:
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

                if note.find("m1") != -1:
                    all_human_dx = prepare_hdx(case_dict["human_dx"])

                    first_using_prompt = Humandx_case_diagnosis_prompt_HDX_shorthand
                    first_fmt_prompt = first_using_prompt.format(
                        text=case_dict["abstract"],
                        all_human_dx=all_human_dx,
                    )

                    diagnosis = gen_diagnosis(first_fmt_prompt, pipe_generator, using_llm, target_results)
                    print(f"Previous Diagnosis: {diagnosis}")

                    # retrieval = prepare_retrieval(case_dict["human_dx"], retriever) # retrieve the triples for the human expert diagnoses
                    retrieval = prepare_retrieval(case_dict["abstract"], retriever) # retrieve the triples for the findings
                    
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"],
                        previous_diagnosis=diagnosis,
                        # all_human_dx=all_human_dx,
                        retrieval=retrieval,
                    )

                    print(fmt_prompt)

                    structured_result = gen_diagnosis(fmt_prompt, pipe_generator, using_llm, target_results)

                if note.find("m2") != -1:
                    all_human_dx = prepare_hdx(case_dict["human_dx"])

                    llm_using_prompt = Humandx_case_diagnosis_prompt_HDX_shorthand
                    llm_fmt_prompt = llm_using_prompt.format(
                        text=case_dict["abstract"],
                        all_human_dx=all_human_dx,
                    )

                    llm_diagnosis = gen_diagnosis(llm_fmt_prompt, pipe_generator, using_llm, target_results)
                    print(f"LLM Diagnosis: {llm_diagnosis}")

                    rag_using_prompt = Humandx_case_diagnosis_prompt_shorthand
                    rag_fmt_prompt = rag_using_prompt.format(
                        text=case_dict["abstract"]
                    )

                    response = query_engine.query(rag_fmt_prompt)
                    parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                    rag_diagnosis = clean_structural_list(parsed_response, target_results)
                    print(f"RAG Diagnosis: {rag_diagnosis}")

                    agg_using_prompt = Humandx_case_diagnosis_prompt_series_aggregation
                    fmt_prompt = agg_using_prompt.format(
                        text=case_dict["abstract"],
                        llm_diagnosis=llm_diagnosis,
                        rag_diagnosis=rag_diagnosis
                    )

                    print(fmt_prompt)

                    structured_result = gen_diagnosis(fmt_prompt, pipe_generator, using_llm, target_results)

                # response = query_engine.query(fmt_prompt)
                # print(f"response: {response}")
                # parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                # structured_result = clean_structural_list(parsed_response, target_results)
                while len(structured_result) == 0:
                    logging.info(f"Response did not contain any {target_results}, Retrying query...")
                    structured_result = gen_diagnosis(fmt_prompt, pipe_generator, using_llm, target_results)
                    # response = query_engine.query(fmt_prompt)
                    # parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results)
                    # structured_result = clean_structural_list(parsed_response, target_results)
                # results.append(f"{diagnosis} | {str(structured_result)}" + "\n")
                results.append(f"{llm_diagnosis} | {rag_diagnosis} | {str(structured_result)}" + "\n")
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
    using_parser = args.using_parser if args.using_parser else "nuparser"
    note = args.note
    quantization = "Q"

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM[using_llm],
        "embed_model_name": EMBED_MODEL[using_embed],
        "storage_dir": f"index/snomed_all_dataset_nodoc_{using_embed}",
        # "storage_dir": f"index/snomed_dataset_nodoc_commandr_hitsnomed", # this is a partial KG indices for testing
        "input_text_dir": EVAL_DATA[eval_dataset],
        # "input_text_dir": "" # uncomment this line to use the custom input text
        "context_window": 32768,
        "max_new_tokens": 2048,
        "case_num":0,
        "verbose": True,
        "quantization": None,
        "similarity_top_k": similarity_top_k,
        "graph_store_query_depth": graph_store_query_depth,
        "retriever_mode": retrieve_mode,
        "test_id": f"_{task}_{eval_dataset}_{using_llm}_{using_embed}_topk{similarity_top_k}_depth{graph_store_query_depth}_{retrieve_mode}_quantization_{quantization}_note_{note}"
    }

    query_engine, pipe_generator, pipe_parser, retriever = rag_init()

    if task == "diagnosis":
        # using_prompt = Humandx_case_diagnosis_prompt_series
        # using_prompt = Humandx_case_diagnosis_prompt_series_retrieval
        # first_using_prompt = Humandx_case_diagnosis_prompt_shorthand
        # first_using_prompt = Humandx_case_diagnosis_prompt_HDX_shorthand

        # prompt_tmpl = PromptTemplate(
        #     using_prompt,
        #     template_var_mappings=prompt_var_mappings
        # )

        for ep in range(1, 11):
            llm_rag_start_diagnosis(test_id=PARAMETERS["test_id"] + f"_ep{ep}",
                                input_text_dir=PARAMETERS["input_text_dir"],
                                # prompt_tmpl=prompt_tmpl,
                                target_results="Diagnosis",
                                query_engine=query_engine,
                                pipe_generator=pipe_generator,
                                pipe_parser=pipe_parser,
                                retriever=retriever,
                                cases=PARAMETERS["case_num"],
                                )
    
    logging.info("Done!")