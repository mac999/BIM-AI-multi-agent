import os
from typing import List

import faiss
import numpy as np
from openai import ChatCompletion, Embedding


class RAGAgent:
    def __init__(self, index_path: str, docs: List[str]):
        self.index = faiss.read_index(index_path)
        self.docs = docs

    def _embed(self, text: str) -> np.ndarray:
        res = Embedding.create(model="text-embedding-ada-002", input=[text])
        return np.array(res["data"][0]["embedding"], dtype="float32")

    def query(self, question: str, top_k: int = 3) -> str:
        vec = self._embed(question)
        D, I = self.index.search(vec.reshape(1, -1), top_k)
        retrieved = "\n".join(self.docs[i] for i in I[0] if i < len(self.docs))
        prompt = f"""사용자 질문: {question}
다음 정보를 활용하여 한국어로 답변하세요:
{retrieved}
"""
        response = ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "문서를 기반으로 답변하세요."},
                     {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message["content"].strip()
