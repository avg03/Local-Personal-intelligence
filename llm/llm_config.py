"""
llm_config.py
HuggingFace open-source model config for dev/testing.
Swap to ChatOllama with "gemma4:e2b" later — summarizer.py doesn't
change at all, it just takes whatever `llm` object you import here.

Two options below: local pipeline (downloads once, runs offline after
that — matches your local-first goal) or hosted Inference API
(no download, fastest for a first test, needs internet + HF token).
Pick ONE, comment out the other.
"""

# ---------------------------------------------------------------
# OPTION A — local pipeline (recommended: matches local-first goal,
# works offline after first download, no token needed)
# ---------------------------------------------------------------
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# pipeline_llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen2.5-0.5B-Instruct",  # small, fast, decent quality for testing
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 300,
#         "temperature": 0.3,
#         "do_sample": True,
#     },
# )
# llm = ChatHuggingFace(llm=pipeline_llm)


# ---------------------------------------------------------------
# OPTION B — hosted Inference API (no local download, needs a free
# HF token from huggingface.co/settings/tokens + internet access)
# ---------------------------------------------------------------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

endpoint_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    huggingfacehub_api_token=load_dotenv().get("HUGGING_FACE_TOKEN"),
    max_new_tokens=300,
    temperature=0.3,
)
llm = ChatHuggingFace(llm=endpoint_llm)