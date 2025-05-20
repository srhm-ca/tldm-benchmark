# Too Long; Didn't Model (TLDM) Benchmark.

TLDM is a large language model (LLM) benchmark for testing long-context understanding drop-off rates — at what point a LLM's understanding of a long-form integrated document begins to degrade. 

## Usage
1. Install the requirements in `requirements.txt`. 
2. Create a GPT-4.1 endpoint through Microsoft Azure (or create a compatible endpoint locally with e.g. `llama.cpp`)
3. Create an endpoint for the model you'd like to test through Hugging Face's inference endpoint API (or create a compatible endpoint with e.g. `llama.cpp`)
4. Plug in the API keys & endpoint URLs into `generate.py` and `test.py` 
5. Define your model's parameters in the `model` match-case in both files; specifically, give it a short key label and define its maximum context length through the `limit` variable
6. Duplicate the existing `blank model` directory and rename it to your model's key name
7. Run!
