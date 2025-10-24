# llm_factory.py
import os
import gc
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # pip install llama-index-embeddings-huggingface
from utils.deepseek_llm import DeepSeek
from typing import Union

# 1. explicitly load llm and embed_model
def build_hf_transformers(
    model_id: str,
    *,
    device_map: str | dict = "auto",
    torch_dtype: torch.dtype | None = torch.float16,
    trust_remote_code: bool = True,
    load_in_8bit: bool | None = None,   # 需要 bitsandbytes
    load_in_4bit: bool | None = None,   # 需要 bitsandbytes
    attn_implementation: str | None = None,  # "flash_attention_2" 等
    token: str | None = None,
    quantization_config: Union[dict, None] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code, token=token)
    model_kwargs = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if load_in_8bit is not None:
        model_kwargs["load_in_8bit"] = load_in_8bit
    if load_in_4bit is not None:
        model_kwargs["load_in_4bit"] = load_in_4bit
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer

# 2. wrap into LlamaIndex HuggingFaceLLM
def wrap_hf_llm(
    model,
    tokenizer,
    *,
    context_window: int = 32768,
    max_new_tokens: int = 256,
    generate_kwargs: dict | None = None,
):
    generate_kwargs = generate_kwargs or {"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True}
    return HuggingFaceLLM(
        model_name=model.config._name_or_path,
        tokenizer_name=tokenizer.name_or_path,
        model=model,
        tokenizer=tokenizer,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generate_kwargs=generate_kwargs,
    )

# 3. provide a factory method to build LLM by name
def build_llm_by_name(
    llm_model_name: str,
    *,
    tokenizer_name: str | None = None,
    context_window: int = 32768,
    max_new_tokens: int = 256,
    output_parser=None,
    api_keys: dict | None = None,
    hf_from_pretrained: bool = False,
    hf_kwargs: dict | None = None,
):
    api_keys = api_keys or {}
    hf_kwargs = hf_kwargs or {}

    if "gpt" in llm_model_name:
        return OpenAI(model=llm_model_name, api_key=api_keys.get("OPENAI_API_KEY"), temperature=0.7)

    if "deepseek" in llm_model_name:
        return DeepSeek(model=llm_model_name, api_key=api_keys.get("DEEPSEEK_API_KEY"), temperature=0.7)

    # Two ways to load HuggingFace models:
    if hf_from_pretrained:
        # more control over model loading, memory management
        model, tokenizer = build_hf_transformers(llm_model_name, token=api_keys.get("HF_TOKEN"), **hf_kwargs)
        return wrap_hf_llm(
            model,
            tokenizer,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
        )
    else:
        # let HuggingFaceLLM handle the model loading, which is more convenient, but less control over memory management
        return HuggingFaceLLM(
            model_name=llm_model_name,
            tokenizer_name=tokenizer_name or llm_model_name,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95, "do_sample": True},
            output_parser=output_parser,
        )

# 4. build embedding model
def build_embed_model(embed_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # LlamaIndex recommends the local embedding wrapper
    return HuggingFaceEmbedding(model_name=embed_model_name)

def configure_global_settings(llm, embed_model):
    Settings.llm = llm
    if embed_model is not None:
        Settings.embed_model = embed_model
    print(f"LLM loaded: {getattr(Settings.llm, 'model_name', getattr(Settings.llm, 'model', 'unknown'))}")
    print(f"embed_model loaded: {getattr(Settings.embed_model, 'model_name', 'unknown')}")
    print("Settings loaded.")

# 5. release model to free up GPU memory, optional
def release_model(model=None, llm_wrapper=None, also_clear_settings=True):
    """
    完整释放模型显存：
    1. Delete the original Transformers model
    2. Delete the HuggingFaceLLM wrapper
    3. Clear the global reference of Settings.llm (optional)
    4. Force garbage collection and CUDA cache clearing

    Args:
        model: The original Transformers model object
        llm_wrapper: HuggingFaceLLM or other LlamaIndex LLM wrapper
        also_clear_settings: Whether to clear Settings.llm at the same time (default True)
    """
    # 1) Delete the original model
    if model is not None:
        try:
            # Optional: Move to CPU before deletion (more thorough in some cases)
            if hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

    # 2) Delete the LlamaIndex wrapper (HuggingFaceLLM, etc.)
    if llm_wrapper is not None:
        # Try to access the internal _model and delete it (some versions use ._model)
        if hasattr(llm_wrapper, "_model"):
            try:
                if hasattr(llm_wrapper._model, "cpu"):
                    llm_wrapper._model.cpu()
                del llm_wrapper._model
            except Exception:
                pass
        if hasattr(llm_wrapper, "_tokenizer"):
            try:
                del llm_wrapper._tokenizer
            except Exception:
                pass
        try:
            del llm_wrapper
        except Exception:
            pass

    # 3) Clear the global reference of Settings (avoid holding old model)
    if also_clear_settings:
        try:
            Settings.llm = None
        except Exception:
            pass

    # 4) Force garbage collection
    gc.collect()

    # 5) Clear CUDA cache (only effective after GPU references are deleted)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: Synchronize all devices (ensure asynchronous operations are complete)
        torch.cuda.synchronize()