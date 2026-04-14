"""
    Module to setup models for RAG.
"""
from typing import Any
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

@staticmethod
def setup_models(config: dict[str, Any]):
    """Configure embedding (CPU) and chat models (GPU via vLLM)"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config["embedding_model"],
        device="cpu",
        trust_remote_code=True
    )

    Settings.llm = OpenAILike(
        model=config["chat_model"],
        api_key=config["vllm_api_key"],
        api_base=config["chat_endpoint"],
        context_window=4096,
        max_tokens=config.get("llm_max_tokens", 512),
        is_chat_model=True,
        is_function_calling_model=False,
        temperature=0.0,
        timeout=config["llm_timeout"],
        additional_kwargs={
            "stop": [
                "\nExplicação:",
                "\nNota:",
                "\nObservação:",
                "Aqui estão os",
                "Os produtos ainda" # Corta a alucinação no ato!
            ]
        }
    )

    Settings.transformations = [
        SentenceSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
    ]
