# Description: This script is used to extract relations from the RAG model using various prompt templates.

from tqdm import tqdm
from rag_prompt_template import *
from rag_util import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


prompt_tmpl = PromptTemplate(
    snomed_relation_extraction_prompt_commandr, 
    template_var_mappings=prompt_var_mappings
)


def query_and_generate_rel(test_id, query_engine, input_text_dir, cases=427):
    logging.info(f"Query Engine: {query_engine}")

    fmt_prompt = prompt_tmpl.format(text="Test")
    logging.info(f"Prompt: {fmt_prompt}")

    with open(input_text_dir) as f:
    # with open("data/chemprot/test.source") as f:
        results = []
        sentences = f.readlines()
        logging.info(f"Experiment ID: {test_id}")
        logging.info(f"Number of sentences: {len(sentences)}; Number of cases for test: {cases}")
        for sentence_id, text in tqdm(enumerate(sentences[:cases])):
            logging.info(f"Processing sentence {sentence_id} / {len(sentences)}")
            # logging.info(f"Text: {text}")
            retry_count = 0

            fmt_prompt = prompt_tmpl.format(
                text=text,
            )
            # print(fmt_prompt)
            response = query_engine.query(fmt_prompt)
            display(Markdown(f"<b>{response}</b>"))
            results.append(extract_triple(str(response)) + "\n")
            print(f"======= Sentence {sentence_id} processed.. ========")

    with open(f"results/rel.hyps_{test_id}", 'w') as f:
        f.writelines(results)
    
    logging.info(f"Results saved to results/rel.hyps_{test_id}")
    return response


if __name__ == '__main__':

    using_llm = "commandr"
    using_embed = "hitsnomed"

    PARAMETERS = {
        "llm_model_name": LLM[using_llm],
        "tokenizer_name": LLM[using_llm],
        "embed_model_name": EMBED_MODEL[using_embed],
        "storage_dir": f"index/snomed_dataset_nodoc_{using_llm}_{using_embed}",
        "input_text_dir": "data/ade1/test.source",
        "case_num": 427,
        "verbose": True,
        "test_id": f"9_snomed_dataset_nodoc_{using_llm}_{using_embed}_simple_as_query_engine_setting2_ade_dec_{using_llm}_{using_embed}"
    }

    logging_setup(log_file=f"logs/{PARAMETERS['test_id']}.log", log_level=logging.INFO)

    # llm, service_context = init_llm_service_context(model_name="HuggingFaceH4/zephyr-7b-beta", tokenizer_name="HuggingFaceH4/zephyr-7b-beta")
    llm, service_context = init_llm_service_context(llm_model_name=PARAMETERS["llm_model_name"], 
                                                    tokenizer_name=PARAMETERS["tokenizer_name"], 
                                                    embed_model_name=PARAMETERS["embed_model_name"]
                                                    )
    # storage_context = init_kg_storage_context(storage_dir="index/snomed_10000_nodoc")
    # storage_context = init_kg_storage_context(storage_dir="index/snomed_25000_nodoc")
    storage_context = init_kg_storage_context(storage_dir=PARAMETERS["storage_dir"])

    query_engine = init_rag_pipeline(llm, service_context, storage_context, verbose=PARAMETERS["verbose"])

    query_and_generate_rel(test_id=PARAMETERS["test_id"], 
                           query_engine=query_engine, 
                           input_text_dir=PARAMETERS["input_text_dir"], 
                           cases=PARAMETERS["case_num"])