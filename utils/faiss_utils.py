import os
from pathlib import Path

import faiss
import numpy as np
from openai import Embedding


def build_faiss_index(doc_dir: str, index_path: str) -> list[str]:
    docs = []
    vectors = []
    for path in Path(doc_dir).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(text)
        res = Embedding.create(model="text-embedding-ada-002", input=[text])
        vec = np.array(res["data"][0]["embedding"], dtype="float32")
        vectors.append(vec)
    if vectors:
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.stack(vectors))
        faiss.write_index(index, index_path)
    return docs
