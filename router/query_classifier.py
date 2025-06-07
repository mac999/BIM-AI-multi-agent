from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModel

LABELS = ["general", "RAG", "web", "IFC"]

_tokenizer = AutoTokenizer.from_pretrained("monologg/distilkobert")
_model = AutoModel.from_pretrained("monologg/distilkobert")


def classify(query: str) -> Literal["general", "RAG", "web", "IFC"]:
    tokens = _tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        outputs = _model(**tokens)
    pooled = outputs.last_hidden_state.mean(dim=1)
    # lightweight keyword-based mapping for demo
    text = query.lower()
    if "ifc" in text:
        return "IFC"
    if any(k in text for k in ["검색", "웹", "인터넷"]):
        return "web"
    if any(k in text for k in ["문서", "자료", "근거"]):
        return "RAG"
    return "general"
