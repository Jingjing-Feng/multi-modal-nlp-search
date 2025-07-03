from threading import Lock

import numpy as np
import rich
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class SBertModel:
    """
    A class representing a Sentence-BERT model.
    """

    _lock = Lock()
    _instance = None

    @classmethod
    def _get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if SBertModel._instance is None:
                    rich.print("[green]Loading SBERT model...[/green]")
                    SBertModel._instance = {
                        "model": SentenceTransformer("all-MiniLM-L6-v2"),
                        "tokenizer": AutoTokenizer.from_pretrained(
                            "sentence-transformers/all-MiniLM-L6-v2",
                            use_fast=True
                        ),
                    }
        return SBertModel._instance

    @classmethod
    def get_embedding(cls, text: str, token_limit=512) -> np.ndarray:
        instance = cls._get_instance()
        model = instance["model"]
        tokenizer = instance["tokenizer"]

        tokenize_text = tokenizer(text)["input_ids"][1:-1]  # remove [CLS] and [SEP]
        chunked_token = [
            tokenize_text[i : i + token_limit]
            for i in range(0, len(tokenize_text), token_limit - 2)
        ]  # -2 to take [CLS] and [SEP] into consideration when tokenizing
        chunked_text = [
            tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunked_token
        ]

        embeddings = model.encode(chunked_text)
        embeddings = np.mean(embeddings, axis=0)
        return embeddings

    @classmethod
    def get_dimension(cls):
        instance = cls._get_instance()
        model = instance["model"]
        return model.encode("random").shape[0]
