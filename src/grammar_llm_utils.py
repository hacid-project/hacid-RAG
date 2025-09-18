from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from grammarllm import generate_grammar_parameters, generate_text, get_parsing_table_and_map_tt, setup_logging, create_prompt, chat_template
import pandas as pd
import re

import torch
from sklearn.model_selection import train_test_split
import numpy as np
import ollama
import gc
from datetime import datetime

setup_logging()

def transform(data_path:str,obj:str):
    df = pd.read_csv(data_path)
    L = []
    for i,row in df.iterrows():
        text = f"<<'{row[obj]}'>>"
        if text not in L:
            L.append(text)
    return L


"""
Two modes:
- MODE="pred": only predicates are constrained within the provided lists, entities can be any alphanumeric string
- MODE="entpred": both entities and predicates are constrained within the provided lists
"""
def init_grammar(MODE="pred"):
    # Define grammar productions
    print(f"Initializing grammar in {MODE} mode...")
    
    predicates_path = "../data/snomed/predicates.csv" #predicates
    predicates = transform(predicates_path,'predicate')

    if MODE=="pred":
        productions = { 'S*': ["[ TS_NT ]"],
                        'TS_NT': ["T_NT TS_TAIL",],  # AT least one triple
                        'TS_TAIL': [", T_NT TS_TAIL", "ε"],  # comma separated triples or empty
                        'T_NT': ["( SUB_NT_L ; PRED_NT ; SUB_NT_R )"], # triple structure
                        'SUB_NT_L': ['alfanum SUB_NT_L', 'ε'],  # left subject is alphanumeric
                        'SUB_NT_R': ['alfanum SUB_NT_R', 'ε'], #left subject is alphanumeric
                        'PRED_NT': predicates, # ensure that predicates belong to the list provided
                        }


        regex_alfanum = re.compile(r"[\u0120a-zA-Z0-9\u2581]+(?![;,\(\)\[\]])", re.UNICODE)
        regex_semicolon = re.compile(r"\u2581?;\u2581?$",re.UNICODE)
        regex_colon = re.compile(r"\u2581?,\u2581?$",re.UNICODE)
        regex_right_round_bracket = re.compile(r"\u2581?\)\u2581?$",re.UNICODE)
        regex_left_round_bracket = re.compile(r"\u2581?\(\u2581?$",re.UNICODE)
        regex_right_square_bracket = re.compile(r"\u2581?\]\u2581?$",re.UNICODE)
        regex_left_square_bracket = re.compile(r"\u2581?\[\u2581?$",re.UNICODE)

        regex_dict = {  'regex_)': regex_right_round_bracket,
                        'regex_(': regex_left_round_bracket,
                        'regex_;':regex_semicolon,
                        'regex_,':regex_colon,
                        'regex_[': regex_left_square_bracket,
                        'regex_]': regex_right_square_bracket,
                        'regex_alfanum': regex_alfanum,
                        
        }

    elif MODE=="entpred":
        entities_path = "../data/snomed/entities.csv" #both sub and obj
        entities = transform(entities_path,'entity')

        productions = { 'S*': ["[ TS_NT ]"],
                'TS_NT': ["T_NT TS_TAIL",],  # AT least one triple
                'TS_TAIL': [", T_NT TS_TAIL", "ε"],  # comma separated triples or empty
                'T_NT': ["( SUB_NT ; PRED_NT ; SUB_NT )"],  # triple structure
                'SUB_NT': entities, # ensure that entities belong to the list provided (nb:since it's a huge list grammar will be ready until 15 minutes, for testing a suggest you to slice the list to a smaller size)
                'PRED_NT': predicates, # ensure that predicates belong to the list provided
                }

        #regex_alfanum = re.compile(r"[a-zA-Z0-9]+")  # es. "abc123"
        regex_semicolon = re.compile(r"\;$")
        regex_colon = re.compile(r"\,$")
        regex_right_round_bracket = re.compile(r"\)$")  # match only ')'
        regex_left_round_bracket = re.compile(r"\($")  # match only '('
        regex_right_square_bracket = re.compile(r"\]$")  # match only ')'
        regex_left_square_bracket = re.compile(r"\[$")  # match only '('
        regex_dict = {  'regex_)': regex_right_round_bracket,
                        'regex_(': regex_left_round_bracket,
                        'regex_;':regex_semicolon,
                        'regex_,':regex_colon,
                        'regex_[': regex_left_square_bracket,
                        'regex_]': regex_right_square_bracket,
                        
        }


    return productions, regex_dict


def generate_text(model, tokenizer, text, logit_processor, streamer, chat_template = None, max_new_tokens=400, do_sample=False, temperature=None, top_p=None, **kwargs):
    """
    Genera testo vincolato dalla grammatica, con configurazione dei parametri di generazione sicura.

    Args:
        model: Il modello pre-addestrato.
        tokenizer: Il tokenizer del modello.
        text: Input text iniziale.
        logit_processor: Processor dei logit basato sulla grammatica.
        streamer: Streamer per l'output live.
        max_new_tokens: Numero massimo di nuovi token da generare.
        do_sample: Se True, abilita la generazione stocastica.
        temperature: Controlla la casualità (usato solo se do_sample=True).
        top_p: Top-p (nucleus sampling), usato solo se do_sample=True.
        **kwargs: Parametri aggiuntivi opzionali per model.generate().
    """
    
    try:
        # TO USE WHEN CREATE PROMPT IS USED AND PROMPT IS A LIST
        if isinstance(text,list):
            if chat_template is None:
                raise ValueError("Chat template must be specified")
            tokenizer.chat_template = chat_template
            tokenized_input = tokenizer.apply_chat_template(text, 
                                                        tokenize=True,
                                                        add_generation_prompt=True,
                                                        return_dict=True,
                                                        return_tensors="pt").to(model.device)
        else:
            tokenized_input = tokenizer(text, return_tensors="pt")

        # Safe defaults
        kwargs.setdefault("num_beams", 1)  # beam search disattivato
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        # Sicurezza num_beams
        if kwargs["num_beams"] != 1:
            logging.warning("⚠️ num_beams > 1 non è compatibile con la generazione vincolata da grammatica. Impostato automaticamente a num_beams=1.")
            kwargs["num_beams"] = 1

        # Sampling parameters
        if do_sample:
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        else:
            # Rimuovi parametri di sampling se presenti
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)

        # Device compatibility
        device = model.device
        input_ids = tokenized_input["input_ids"].to(device)
        if input_ids.device != model.device:
            logging.warning("Errore: gli 'input_ids' sono sulla device {input_ids.device}, mentre il modello è sulla device {model.device}. Spostando 'input_ids' sulla stessa device del modello.")
        
        attention_mask = tokenized_input["attention_mask"].to(device)
        if attention_mask.device != model.device:
            logging.warning(f"Errore: l'attention_mask è sulla device {attention_mask.device}, mentre il modello è sulla device {model.device}. Spostando 'attention_mask' sulla stessa device del modello.")
        

        start = input_ids.shape[1]

        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            logits_processor=[logit_processor],
            **kwargs
        )

        answer = tokenizer.decode(output[0][start:], skip_special_tokens=True)

        return answer

    except Exception as e:
        raise RuntimeError(f"Errore nella generazione del testo: {e}")