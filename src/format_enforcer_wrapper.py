from llama_index.program.lmformatenforcer import LMFormatEnforcerPydanticProgram
from pydantic import BaseModel
from typing import List
import re

# 定义输出格式
class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

class TripleResponse(BaseModel):
    triples: List[Triple]

class EntityResponse(BaseModel):
    entities: List[str]

def remove_incomplete_triples(response: str) -> str:
    """
    Extract full triples of the form (subject, predicate, object) and remove any incomplete ones.
    """
    pattern = re.compile(r"\(.*?,.*?,.*?\)")
    matches = pattern.findall(response)
    return " ".join(matches) if matches else response

def create_hybrid_triple_engine(original_query_engine, llm):
    # keep the original query engine for getting the full response
    # original_query_engine = kg_index.as_query_engine(llm=llm)
    
    # create the format enforcer program
    program = LMFormatEnforcerPydanticProgram(
        output_cls=TripleResponse,
        prompt_template_str="""
        extract triples according to the following information:

        Original response: {original_response}
        Original query: {query}

        Return only the triples in the specified JSON format.

        {json_schema}
        
        """,
        llm=llm,
        verbose=True
    )
    
    def hybrid_query(prompt):
        # retrieve the original response first
        original_response = original_query_engine.query(prompt)
        # print(f"Original response: {original_response}")

        # remove incomplete triples
        orig_str = str(original_response)
        cleaned = remove_incomplete_triples(orig_str)
        # print(f"Cleaned response: {cleaned}")

        # use format enforcer to reformat
        formatted_result = program(
            original_response=cleaned,
            query=prompt
        )
        
        return formatted_result.triples
    
    return hybrid_query

def create_hybrid_entity_extractor(original_query_engine, llm):
    """Create a hybrid entity extraction engine using format enforcer."""

    # create the format enforcer program
    program = LMFormatEnforcerPydanticProgram(
        output_cls=EntityResponse,
        prompt_template_str="""
        extract entities according to the following information:

        Original response: {original_response}
        Original query: {query}

        Return only the entity names in the specified JSON format.

        {json_schema}

        """,
        llm=llm,
        verbose=True
    )
    
    def extract_entities(prompt):
        # Stage one: get the original response
        original_response = original_query_engine.query(prompt)
        # print(f"Original response: {original_response}")

        # Stage two: use format enforcer to extract entities
        result = program(
            original_response=str(original_response),
            query=prompt
        )
        
        return result.entities
    
    return extract_entities