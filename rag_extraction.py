# Description: This script is used to extract relations from the RAG model using various prompt templates.

from tqdm import tqdm
from rag_prompt_template import *
from rag_util import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--using_llm', type=str, required=True, help='The LLM model to use')
    parser.add_argument('--using_embed', type=str, required=True, help='The embedding model to use')
    parser.add_argument('--eval_dataset', type=str, required=False, help='The dataset to evaluate')
    parser.add_argument('--topk', type=int, required=True, help='The top k to retrieve')
    parser.add_argument('--depth', type=int, required=True, help='The depth to query')
    parser.add_argument('--retrieve_mode', type=str, required=True, help='The retrieve mode to use')
    parser.add_argument('--using_extractor', type=str, required=False, help='The helper extractor to use')
    
    args = parser.parse_args()
    return args

def entity_extraction(text, pipe_extractor, using_extractor):
    if using_extractor == "mistralsmall":
        entities = entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)[0]["generated_text"].split("Entities: ")[1].replace("[][/INST]", "").strip()
    elif using_extractor == "solar":
        entities = entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)
        if "assistant\Entities:" in entities:
            entities = entities[0]["generated_text"].split("assistant\Entities:")[1].strip()
        else:
            entities = entities[0]["generated_text"].split("Entities:")[1].strip()
    elif using_extractor == "gpt-4o-mini":
        entities = entity_extractor(text=text, pipe=pipe_extractor, using_extractor=using_extractor)
    return entities

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
            results.append(extract_triple(str(response)) + "\n")
            print(f"======= Sentence {sentence_id} processed.. ========")

    with open(f"results/{test_id}", 'w') as f:
        f.writelines(results)
    
    logging.info(f"Results saved to results/rel.hyps_{test_id}")


if __name__ == '__main__':

    args = parse_arguments()
    using_llm = args.using_llm if args.using_llm else "mistralsmall"
    using_embed = args.using_embed if args.using_embed else "hitsnomed"
    eval_dataset = args.eval_dataset if args.eval_dataset else "BC5CDR"
    similarity_top_k = args.topk if args.topk else 30
    graph_store_query_depth = args.depth if args.depth else 5
    retrieve_mode = args.retrieve_mode if args.retrieve_mode else "hybrid"
    using_extractor = args.using_extractor

    if using_extractor is None: # w/o extractor mode
        using_prompt = BC5CDR_extraction_prompt
    else: # with extractor mode
        using_prompt = BC5CDR_extraction_prompt_with_entities

    prompt_tmpl = PromptTemplate(
        using_prompt,
        template_var_mappings=prompt_var_mappings
    )

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM[using_llm],
        "embed_model_name": EMBED_MODEL[using_embed],
        "storage_dir": f"index/snomed_all_dataset_nodoc_{using_embed}",
        # "input_text_dir": "data/BioNEL_datasets/NCBIdevelopset.json", 
        "input_text_dir": "data/BioNEL_datasets/BC5CDR_gold_all.json",
        "context_window": 32768,
        "max_new_tokens": 2048,
        "case_num":50,
        "verbose": True,
        "similarity_top_k": similarity_top_k,
        "graph_store_query_depth": graph_store_query_depth,
        "retriever_mode": retrieve_mode,
        "test_id": f"41_snomed_dataset_nodoc_simple_as_query_engine_{eval_dataset}_{using_llm}_{using_embed}_topk{similarity_top_k}_depth{graph_store_query_depth}_{retrieve_mode}_extractor_{using_extractor}"
    }

    logging_setup(log_file=f"logs/{PARAMETERS['test_id']}.log", log_level=logging.INFO)
    logging.info(f"PARAMETERS: {PARAMETERS}")
    logging.info(f"Prompt: {using_prompt}")

    llm = init_llm_service_context(llm_model_name=PARAMETERS["llm_model_name"], 
                                    tokenizer_name=PARAMETERS["tokenizer_name"], 
                                    embed_model_name=PARAMETERS["embed_model_name"],
                                    context_window=PARAMETERS["context_window"],
                                    max_new_tokens=PARAMETERS["max_new_tokens"]
                                )
    
    if using_extractor is not None:
        pipe_extractor = init_llm_pipeline(LLM[using_extractor])

    # storage_context = init_kg_storage_context(storage_dir="index/snomed_10000_nodoc")
    # storage_context = init_kg_storage_context(storage_dir="index/snomed_25000_nodoc")
    kg_index = init_kg_storage_context(llm, storage_dir=PARAMETERS["storage_dir"])
    # kg_index = init_kg_storage_context(llm, storage_dir="index/snomed_dataset_nodoc_commandr_minilml6v2")

    query_engine = init_rag_pipeline(kg_index, 
                                     similarity_top_k=PARAMETERS["similarity_top_k"], 
                                     graph_store_query_depth=PARAMETERS["graph_store_query_depth"], 
                                     include_text=False, 
                                     retriever_mode=PARAMETERS["retriever_mode"],
                                     verbose=PARAMETERS["verbose"])

    query_and_generate_rel(test_id=PARAMETERS["test_id"], 
                           query_engine=query_engine, 
                           input_text_dir=PARAMETERS["input_text_dir"], 
                           cases=PARAMETERS["case_num"])
    
    logging.info("Done!")