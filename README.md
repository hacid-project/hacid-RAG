> [!NOTE]
> The current repository version is for test usage and it's under update.

### Quick Usage

1. Download the .zip file of the indexed knowledge graph [here](https://drive.google.com/drive/folders/1FuU0hpjdNEFad9uB6rIP38F5Qygr7Dme).
- **snomed_all_dataset_nodoc_hitsnomed.zip:** The indices of full knowledge graph embedded by [hitsnomed](https://huggingface.co/Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT)
- **snomed_all_dataset_nodoc_minilml12v2.zip:** The indices of full knowledge graph embedded by [minilml12v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- **snomed_partial_dataset_nodoc_minilml6v2.zip:** The indices of **partial** knowledge graph embedded by [minilml6v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

2. Download the LLM and embedding model (recommendations: [mistral-small](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409) and [hitsnomed](https://huggingface.co/Hierarchy-Transformers/HiT-MiniLM-L12-SnomedCT) ). Place them in a new folder and check the paths in the [rag_util.py](https://github.com/hacid-project/hacid-RAG/blob/master/rag_util.py)

`$ LLM = {}` and 
`EMBED_MODEL = {}`

3. Unzip the file and put it into the "index" folder. (If there is no "index" folder, create a new one under the root directory.)

4. Run the [rag_playground.ipynb](https://github.com/hacid-project/hacid-RAG/blob/master/rag_playground.ipynb) for test. This notebook contains several working scenarios for one sample input. For testing file input, run [rag_extraction.py](https://github.com/hacid-project/hacid-RAG/blob/master/rag_extraction.py).

                
