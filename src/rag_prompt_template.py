relation_list = [
    "temporally follows",
    "after",
    "due to",
    "has realization",
    "associated with",
    "has definitional manifestation",
    "associated finding",
    "associated aetiologic finding",
    "associated etiologic finding",
    "interprets",
    "associated morphology",
    "causative agent",
    "course",
    "finding site",
    "temporally related to",
    "pathological process",
    "direct morphology",
    "is modification of",
    "measures",
    "direct substance",
    "has active ingredient",
    "using",
    "part of"
]

# ======================================
# snomed concepts extraction zbeta
# ======================================
snomed_extraction_prompt_zbeta = """\
Extract the most likely concepts with type from the given context with the format of "(concept; type)".\

Here is the context: {text}.\

concepts:\

\

Note: Only the SNOMED CT concepts are allowed; Compound phrase first; Remove repetition; Only output concepts and types.\

"""


# ======================================
# snomed concepts extraction
# ======================================
snomed_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT triplets from the given context with the format of (concept ; is ; type).\

Here is the optional type list: [disorder, clinical finding, substance, morphologically abnormal structures, organism].\

The steps are as follows:\
1. extract the concept from the given context sentence, using the retrieved sub-graph.
2. select the most likely type from the list for the extracted concept.
3. output the triplets in the format of (concept ; is ; type) strictly.\
\

triplets:\

\

Note: Only output the triplets.\

"""

# ======================================
# snomed relation extraction
# ======================================
snomed_relation_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT triples from the given context with the format of (concept 1 ; relation ; concept 2).\

Here is the optional relation list: [temporally follows, after, due to, has realization, associated with, has definitional manifestation, 
associated finding, associated aetiologic finding, associated etiologic finding, interprets, associated morphology, causative agent, course, 
finding site, temporally related to, pathological process, direct morphology, is modification of, measures, direct substance, has active ingredient, using, part of].\

The steps are as follows:\
1. extract the concept 1 and concept 2 from the given context sentence, using the retrieved sub-graph.
2. select ONE most likely relation from the list for the extracted concepts.
3. output the triplets in the format of (concept 1 ; relation ; concept 2) strictly.\
\

Provide your answer as follows:

Answer:::
Triples: (The extracted triples)\
Answer End:::\

You MUST provide values for 'Triples:' in your answer.\

"""


# ======================================
# snomed description generation
# ======================================
snomed_description_generation_prompt = """\
Here is the context: {text}.\

Here is the optional relation list: [temporally follows, after, due to, has realization, associated with, has definitional manifestation, 
associated finding, associated aetiologic finding, associated etiologic finding, interprets, associated morphology, causative agent, course, 
finding site, temporally related to, pathological process, direct morphology, is modification of, measures, direct substance, has active ingredient, using, part of].\

Task: Generate the SNOMED CT descriptions for the given concept.

The steps are as follows:
1. extract the concept 1 from the given context sentence, using the retrieved sub-graph.
2. generate the concept 2 that can describe the concept 1, and select ONE most likely relation from the list for the concept 1.
3. output (concept 1 ; relation ; concept 2) strictly as one generated description.
4. Each extracted concept could have multiple descriptions.\

Provide your answer as follows:

Answer:::
[Extracted Concept] (The generated description) (The generated description)\
Answer End:::\

You MUST provide values for 'Extracted Concept' and 'The generated descriptions' in your answer.\

"""


# ======================================
# BC5CDR entity-type extraction
# ======================================
BC5CDR_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the entity-type pairs from the given context with the format of (entity ; type).\

Here is the type list: [Disorder, Substance].\

The steps are as follows:\
1. extract the entity from the given context abstract, using the retrieved sub-graph.
2. select ONE most likely type from the list for the extracted entity.
3. output the pairs in the format of (entity ; type) strictly.
4. repeat the step 1 to step 3.\
\

Provide your answer as follows:

Answer:::
Pairs: (All extracted pairs)\
Answer End:::\

Requirements:\
You MUST provide values for 'Pairs:' in your answer. \
ONLY use the type in the type list: [Disorder, Substance].\
ONLY output valid entity-type pairs without any reasoning.

"""


# ======================================
# BC5CDR entity-type extraction with additional entities
# ======================================
BC5CDR_extraction_prompt_with_entities = """\
You are a medical professional working in a hospital. You have been given a medical abstract, a list of entities, and a list of types. Your task is to link the entities to the most likely type from the type list.

Here is the abstract: {text}.\

Here is the type list: [Disorder, Substance].\

Here is the list of entities for consideration: {entities}.\

Task: link the entity and the type and output entity-type pairs with the format of (entity ; type).\

The steps are as follows:
1. for each entity in {entities}, link it to the most likely type from the type list. if you cannot find a suitable type, ignore the entity.
2. if you find more entities in the abstract, extract them and link them to the most likely type.
3. output the pairs in the format of (entity ; type) strictly.\
\

Provide your answer as follows:

Answer:::
Pairs: (entity ; type)
Answer End:::\

Requirements:
You MUST provide values for 'Pairs:' in your answer. 
ONLY use the type in the type list: [Disorder, Substance]. 
ONLY output valid entity-type pairs without any reasoning. 

"""


# ======================================
# NCBIdevelopset entity-type extraction
# ======================================
NCBIdevelopset_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the entity-type pairs from the given context with the format of (entity ; type).\

Here is the type list: ['DiseaseClass', 'SpecificDisease', 'Modifier', 'CompositeMention'].\

The steps are as follows:\
1. extract the entity from the given context abstract, using the retrieved sub-graph.
2. select ONE most likely type from the list for the extracted entity.
3. output the pairs in the format of (entity ; type) strictly.
4. repeat the step 1 to step 3.\
\

Provide your answer as follows:

Answer:::
Pairs: (All extracted pairs)\
Answer End:::\

Requirements:\
You MUST provide values for 'Pairs:' in your answer. \
ONLY use the type in the type list: ['DiseaseClass', 'SpecificDisease', 'Modifier', 'CompositeMention'].\
Extract as many valid entity-type pairs as possible from the given context abstract.\

"""


# ======================================
# MIMIC-IV snomed entity extraction
# ======================================
MIMICIV_entity_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT concepts from the given context.\

The steps are as follows:\
1. extract the concepts from the given context sentence, using the retrieved triplets.
2. there may be abbreviations or acronyms in the context, extract them as concepts as well if they are related to the concepts.
3. output the concepts in a list [] strictly, each concept is separated by a comma.\
\

Provide your answer as follows:

Answer:::
Concepts: [] \
Answer End:::\

Requirements:\
You MUST provide values for 'Concepts:' in your answer. \
ONLY extract concepts, DO NOT include the type of the concept, reasoning, or any other information. \
DO NOT include mark numbers or ordinal numbers in your answer. \
Extract as many unique concepts as possible from the given context. \

"""

# ======================================
# IoU MIMIC-IV snomed entity extraction
# ======================================
finetuned_MIMICIV_entity_extraction_prompt = """\
Extract the mentioned SNOMED CT concepts from the given discharge note.

Here is the desired types of the concepts: [finding, disorder, procedure, regimen/therapy, morphologic abnormality, body structure, cell structure]

Here is the discharge note: {text}.

"""


IoU_MIMICIV_entity_extraction_prompt = """\
Task: Extract the concepts from the given discharge note.\

Here is the desired types of the concepts: [finding, disorder, procedure, regimen/therapy, morphologic abnormality, body structure, cell structure]

The steps are as follows:\
1. extract the concepts from the given discharge note that belong to one of those types, using the retrieved triples.
2. there may be abbreviations or acronyms in the context, extract them as concepts as well if they are related to the concepts.
3. output the concepts in a list [] strictly, each concept is separated by a comma.\
\

Provide your answer as follows:

Answer:::
Concepts: [] \
Answer End:::\

Requirements:\
You MUST provide values for 'Concepts:' in your answer. \
ONLY extract concepts, DO NOT include the type of the concept, reasoning, or any other information. \
DO NOT include mark numbers or ordinal numbers in your answer. \
Extract as many unique concepts as possible from the given context. \

Here is the discharge note: {text}.\

"""


IoU_MIMICIV_entity_extraction_prompt_with_entities = """\
Task: Extract the concepts from the given discharge note.\

Here is the desired types of the concepts: [finding, disorder, procedure, regimen/therapy, morphologic abnormality, body structure, cell structure]

Here is the list of concepts for consideration: {entities}.

The steps are as follows:\
1. for each concept in {entities}, keep it if it belongs to one of the types. if you cannot find a suitable type, ignore the concept.
2. if you find more concepts in the discharge note, extract them and keep them if they belong to one of the types.
3. there may be abbreviations or acronyms in the context, extract them as concepts as well if they are related to the concepts.
4. output the concepts in a list [] strictly, each concept is separated by a comma.\
\

Provide your answer as follows:

Answer:::
Concepts: [] \
Answer End:::\

Requirements:\
You MUST provide values for 'Concepts:' in your answer. \
ONLY extract concepts, DO NOT include the type of the concept, reasoning, or any other information. \
DO NOT include mark numbers or ordinal numbers in your answer. \
Extract as many unique concepts as possible from the given context. \

Here is the discharge note: {text}.\

"""


# ======================================
# MIMIC-IV deepseek snomed entity extraction
# ======================================
MIMICIV_deepseek_entity_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT concepts from the given context.\

The steps are as follows:\
1. extract the concepts from the given context sentence, using the retrieved triplets.
2. there may be abbreviations or acronyms in the context, extract them as concepts as well if they are related to the concepts.
3. output the concepts in a list [] strictly, each concept is separated by a comma.\
\

Provide your answer as follows:

Answer:::
Concepts: [] \
Answer End:::\

Requirements:\
You MUST provide values for 'Concepts:' in your answer. \
ONLY extract concepts, DO NOT include the type of the concept, reasoning, or any other information. \
DO NOT include the chain of thought in the answer. \
Extract as many unique concepts as possible from the given context. \

"""

# ======================================
# MIMIC-IV entity-type extraction
# ======================================
MIMICIV_extraction_prompt = """\

Here is the context: {text}.\

Task: Extract the entity-type pairs from the given context with the format of (entity ; type).\

Here is the type list: [finding, disorder, procedure, regime/therapy, morphologic abnormality, body structure, cell structure].\

The steps are as follows:\
1. extract the entity from the given context abstract, using the retrieved sub-graph.
2. select ONE most likely type from the list for the extracted entity.
3. output the pairs in the format of (entity ; type) strictly.
4. repeat the step 1 to step 3.\
\
Provide your answer as follows:

Answer:::
Pairs: (All extracted pairs)\
Answer End:::\
\

Requirements:\
You MUST provide values for 'Pairs:' in your answer. \
output the pairs in the format of (entity ; type) strictly. \

"""


# ======================================
# MIMIC-IV entity-type extraction with additional entities
# ======================================
MIMICIV_extraction_prompt_with_entities = """\
You are a medical professional working in a hospital. You have been given a discharge note, a list of entities, and a list of types. Your task is to link the entities to the most likely type from the type list.

Here is the abstract: {text}.\

Here is the type list: [finding, disorder, procedure, regime/therapy, morphologic abnormality, body structure, cell structure].\

Here is the list of entities for consideration: {entities}.\

Task: link the entity and the type and output entity-type pairs with the format of (entity ; type).\

The steps are as follows:
1. for each entity in {entities}, link it to the most likely type from the type list. if you cannot find a suitable type, ignore the entity.
2. if you find more entities in the abstract, extract them and link them to the most likely type.
3. output the pairs in the format of (entity ; type) strictly.\
\

Provide your answer as follows:

Answer:::
Pairs: (entity ; type)
Answer End:::\

Requirements:
You MUST provide values for 'Pairs:' in your answer. 
ONLY output valid entity-type pairs without any reasoning. 

"""

# ======================================
# Pubmed snomed triple extraction
# ======================================
Pubmed_snomed_triple_extraction_prompt = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT triples from the given context with the format of (concept 1 ; relation ; concept 2).\

Here is the optional relation list: [temporally follows, after, due to, has realization, associated with, has definitional manifestation,
associated finding, associated aetiologic finding, associated etiologic finding, interprets, associated morphology, causative agent, course,
finding site, temporally related to, pathological process, direct morphology, is modification of, measures, direct substance, has active ingredient, using, part of].\

The steps are as follows:\
1. extract the concept 1 and concept 2 from the given context sentence, using the retrieved sub-graph.
2. select ONE most likely relation from the list for the extracted concepts.
3. output the triples in the format of (concept 1 ; relation ; concept 2) strictly.\
\

Provide your answer as follows:

Answer:::
Triples: (The extracted triples)\
Answer End:::\

Requirements:\
You MUST provide values for 'Triples:' in your answer.\
ONLY output the triples without any other information.\
"""

Pubmed_snomed_triple_extraction_prompt_with_entities = """\
Here is the context: {text}.\

Task: Extract the SNOMED CT triples from the given context with the format of (concept 1 ; relation ; concept 2).\

Here is the optional relation list: [temporally follows, after, due to, has realization, associated with, has definitional manifestation,
associated finding, associated aetiologic finding, associated etiologic finding, interprets, associated morphology, causative agent, course,
finding site, temporally related to, pathological process, direct morphology, is modification of, measures, direct substance, has active ingredient, using, part of].\

Here is the list of entities for consideration: {entities}.\

The steps are as follows:\
1. extract the concept 1 and concept 2 from the given context sentence, using the retrieved sub-graph.
2. select ONE most likely relation from the list for the extracted concepts.
3. output the triples in the format of (concept 1 ; relation ; concept 2) strictly.\

Provide your answer as follows:

Answer:::
Triples: (The extracted triples)\
Answer End:::\

Requirements:\
You MUST provide values for 'Triples:' in your answer.\
ONLY output the triples without any other information.\
"""

prompt_var_mappings = {"text": "text"}


# ======================================
# Human-dx snomed entity extraction
# ======================================
Humandx_entity_extraction_prompt = """\
Here is the context: {text}.

Task: Extract the SNOMED CT concepts from the given context.

The steps are as follows:
1. extract the concepts from the given context sentence, using the retrieved triplets.
2. there may be abbreviations or acronyms in the context, extract them as concepts as well if they are related to the concepts.
3. output the concepts in a list [] strictly, each concept is separated by a comma.


Provide your answer as follows:

Answer:::
Concepts: []
Answer End:::

Requirements:
You MUST provide values for 'Concepts:' in your answer. 
ONLY extract concepts, DO NOT include the type of the concept, reasoning, or any other information. 
DO NOT include mark numbers or ordinal numbers in your answer. 
Extract as many unique concepts as possible from the given context. 

"""


Humandx_case_diagnosis_prompt = """\
Case findings: {text}

read given case findings and analyse them, then provide up to 10 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. 
Be as concise as possible, no need to be polite.

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response." 

"""

Humandx_case_diagnosis_prompt_HDX = """
Case findings: {text}

read given case findings and analyse them, then provide up to 10 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. 
Be as concise as possible, no need to be polite.

Here also some diagnosis from human experts for consideration:
{all_human_dx}

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

Humandx_case_diagnosis_prompt_5 = """\
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. 
Be as concise as possible, no need to be polite.

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response." 

"""

Humandx_case_diagnosis_prompt_HDX_5 = """
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. 
Be as concise as possible, no need to be polite.

Here also some diagnosis from human experts for consideration:
{all_human_dx}

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

# ======================================

Humandx_case_diagnosis_prompt_shorthand = """\
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response." 

"""

Humandx_case_diagnosis_prompt_HDX_shorthand = """
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Here also some diagnosis from human experts for consideration:
{all_human_dx}

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

Humandx_case_diagnosis_prompt_HDX_Retrieval_shorthand = """
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Here are some retrieved knowledge from SNOMED-CT for consideration:
{retrieval}

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

Humandx_case_diagnosis_prompt_HDX_SPARQLKG_shorthand = """
Case findings: {text}

read given case findings and analyse them, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Here also some diagnosis from human experts for consideration:
{all_human_dx}

Here are some knowledge retrieved by SPARQL from SNOMED-CT for consideration:
{SPARQL_KG}

Provide your answer as follows:

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

# ======================================
# Series connection of LLM and RAG
# ======================================
Humandx_case_diagnosis_prompt_series = """
Case findings: {text}

Regarding the given case findings, the previous diagnosis is {previous_diagnosis}.

Now here is the diagnosis from human experts for consideration:
{all_human_dx}

Refine the previous diagnosis based on the human experts' diagnosis, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

Humandx_case_diagnosis_prompt_series_retrieval = """
Case findings: {text}

Regarding the given case findings, the previous diagnosis is {previous_diagnosis}.

Here are some retrieved knowledge from SNOMED-CT for consideration:
{retrieval}

Refine the previous diagnosis based on the retrieved relevant knowledge, then provide up to 5 probable diagnostic solutions. no explanation, no recapitulation of the case information or task. 
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses. 
Be as concise as possible, no need to be polite.

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""

Humandx_case_diagnosis_prompt_series_aggregation = """
Case findings: {text}

Regarding the given case findings, the previous diagnosis are {llm_diagnosis} and {rag_diagnosis}.

Now regarding the case findings, aggregate the two previous diagnosis and provide up to 5 probable final diagnostic solutions. no explanation, no recapitulation of the case information or task.
all diagnostic solutions are sorted by probability, most probable first. In your answer use common shorthand non-abbreviated diagnoses.
Be as concise as possible, no need to be polite.

Answer:::
Diagnosis: (most probable first)
Answer End:::

You MUST provide values for 'Diagnosis' in your answer.
Do not provide any other information in your response.

"""
