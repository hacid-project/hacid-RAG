# ========================================
# mistralsmall RAG + mimic entity
# ========================================
python rag_extraction.py --using_llm mistralsmall --using_embed hitsnomed --eval_dataset mimicivchunk --topk 50 --depth 5 --retrieve_mode embedding
