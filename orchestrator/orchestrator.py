from typing import Dict

from agents import general_agent, rag_agent, web_agent, ifc_agent
from router import query_classifier


class Orchestrator:
    def __init__(self, rag: rag_agent.RAGAgent):
        self.rag = rag

    def handle_question(self, question: str, ifc_path: str | None = None) -> Dict[str, str]:
        label = query_classifier.classify(question)
        if label == "general":
            answer = general_agent.answer_question(question)
        elif label == "RAG":
            answer = self.rag.query(question)
        elif label == "web":
            answer = web_agent.search_and_summarize(question)
        elif label == "IFC" and ifc_path:
            answer = ifc_agent.answer_question(ifc_path, question)
        else:
            answer = general_agent.answer_question(question)
        return {"route": label, "answer": answer}
