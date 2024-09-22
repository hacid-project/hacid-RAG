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
# snomed concepts extraction commandr
# ======================================
snomed_extraction_prompt_commandr = """\
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
# snomed relation extraction commandr
# ======================================
snomed_relation_extraction_prompt_commandr = """\
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
# snomed description generation commandr
# ======================================
snomed_description_generation_prompt_commandr = """\
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


prompt_var_mappings = {"text": "text"}