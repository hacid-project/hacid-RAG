# Description: This script is used to extract relations from the RAG model using various prompt templates.

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
    parser.add_argument('--eval_dataset', type=str, required=True, choices=['BC5CDR', 'NCBI', 'MIMICIV', 'pubmed', 'humandx'], help='The evaluation dataset to use')
    parser.add_argument('--using_parser', type=str, required=False, help='The parser to use')
    parser.add_argument('--note', type=str, required=False, help='Any additional note, e.g. with or without human diagnosis')
    
    args = parser.parse_args()
    return args


def llm_init():
    logging_setup(log_file=f"logs/{PARAMETERS['test_id']}.log", log_level=logging.INFO)
    logging.info(f"PARAMETERS: {PARAMETERS}")
    # logging.info(f"Prompt: {using_prompt}")

    pipe_generator = init_llm_pipeline(LLM[using_llm], 
                                    #    quantization_config=None
                                       )
    pipe_parser = init_llm_pipeline(LLM[using_parser], quantization_config=None) if using_parser is not None else None

    retriever = None

    if note.find("Retrieval") != -1:
        llm = init_llm_service_context(llm_model_name=PARAMETERS["llm_model_name"], 
                                        tokenizer_name=PARAMETERS["tokenizer_name"], 
                                        embed_model_name=EMBED_MODEL["hitsnomed"],
                                        context_window=PARAMETERS["context_window"],
                                        max_new_tokens=PARAMETERS["max_new_tokens"],
                                        quantization_config=quantization_config,
                                        )
        kg_index = init_kg_storage_context(llm, storage_dir=f"index/snomed_all_dataset_nodoc_hitsnomed")
        # kg_index = init_kg_storage_context(llm, storage_dir=f"index/snomed_dataset_nodoc_commandr_hitsnomed") # this is a partial index for testing
        retriever = init_retriever(kg_index=kg_index,
                                    similarity_top_k=30,
                                    graph_store_query_depth=5,
                                    verbose=True,
                                    retriever_mode="hybrid",
                                    )
        
    return pipe_generator, pipe_parser, retriever

# start query and extract information
def llm_start(test_id, 
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
                while len(structured_result) == 0:
                    logging.info(f"Response did not contain any {target_results}, Retrying query...")
                    response = query_engine.query(fmt_prompt)
                    parsed_response = llm_parser(response, pipe_parser, "nuparser", target_results).strip()
                    structured_result = clean_structural_list(parsed_response, target_results)
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


def llm_start_diagnosis(test_id, 
                        input_text_dir, 
                        pipe_generator,
                        prompt_tmpl, 
                        target_results, 
                        pipe_parser=None, 
                        retriever=None,
                        cases=0
                        ):
    logging.info(f"LLM Generator: {pipe_generator}")
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
                    # print(f"Prompt: {fmt_prompt}")

                elif note.find("Retrieval") != -1: # if the note contains Retrieval, use the Retrieval prompt
                    # all_human_dx = prepare_hdx(case_dict["human_dx"])
                    retrieval = prepare_retrieval(case_dict["human_dx"], retriever)
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"],
                        # all_human_dx=all_human_dx,
                        retrieval=retrieval
                    )

                    print(f"Retrieval Prompt: {fmt_prompt}")

                elif note.find("SPARQLKG") != -1: # if the note contains SPARQLKG, use the SPARQLKG prompt
                    SPARQL_KG = ", ".join(case_dict["SPARQL_KG"])
                    all_human_dx = prepare_hdx(case_dict["human_dx"]) 
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"],
                        all_human_dx=all_human_dx,
                        SPARQL_KG=SPARQL_KG
                    )

                    print(f"SPARQLKG Prompt: {fmt_prompt}")

                else:
                    fmt_prompt = prompt_tmpl.format(
                        text=case_dict["abstract"]
                    )

                # print(fmt_prompt)
                response = llm_generation(fmt_prompt, pipe_generator, using_llm)
                # print(f"response: {response}")
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
                while len(structured_result) == 0:
                    logging.info(f"Response did not contain any {target_results}, Retrying query...")
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
    eval_dataset = args.eval_dataset if args.eval_dataset else "BC5CDR"
    using_parser = args.using_parser if args.using_parser else "nuparser"
    note = args.note
    quantization = "Q"

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM[using_llm],
        "input_text_dir": EVAL_DATA[eval_dataset],
        # "input_text_dir": f"", # uncomment this line to use the custom input text
        "context_window": 32768,
        "max_new_tokens": 2048,
        "case_num":50,
        "quantization": quantization,
        "test_id": f"_{task}_{eval_dataset}_{using_llm}_quantization_{quantization}_note_{note}"
    }

    pipe_generator, pipe_parser, retriever = llm_init()

    if task == "diagnosis":
        if note.find("HDX") != -1: # if the note contains HDX, use the HDX prompt
            using_prompt = Humandx_case_diagnosis_prompt_HDX_shorthand
        elif note.find("Retrieval") != -1: # if the note contains Retrieval, use the Retrieval prompt
            using_prompt = Humandx_case_diagnosis_prompt_HDX_Retrieval_shorthand
        elif note.find("SPARQLKG") != -1: # if the note contains SPARQLKG, use the SPARQLKG prompt
            using_prompt = Humandx_case_diagnosis_prompt_HDX_SPARQLKG_shorthand
        else:
            using_prompt = Humandx_case_diagnosis_prompt_shorthand

        prompt_tmpl = PromptTemplate(
            using_prompt,
            template_var_mappings=prompt_var_mappings
        )

        for ep in range(1, 8):
            llm_start_diagnosis(test_id=PARAMETERS["test_id"] + f"_ep{ep}",
                                input_text_dir=PARAMETERS["input_text_dir"],
                                pipe_generator=pipe_generator,
                                prompt_tmpl=prompt_tmpl,
                                target_results="Diagnosis",
                                pipe_parser=pipe_parser,
                                retriever=retriever,
                                cases=PARAMETERS["case_num"],
                                )
    
    else:
        using_prompt = Humandx_case_diagnosis_prompt
        prompt_tmpl = PromptTemplate(
            using_prompt,
            template_var_mappings=prompt_var_mappings
        )
        llm_start(test_id=PARAMETERS["test_id"],
                    input_text_dir=PARAMETERS["input_text_dir"],
                    query_engine=pipe_generator,
                    prompt_tmpl=prompt_tmpl,
                    target_results="Concepts",
                    pipe_parser=pipe_parser,
                    cases=PARAMETERS["case_num"],
                    )
    
    logging.info("Done!")