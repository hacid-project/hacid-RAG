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
Extract as many valid entity-type pairs as possible from the given context abstract.\

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


prompt_var_mappings = {"text": "text"}