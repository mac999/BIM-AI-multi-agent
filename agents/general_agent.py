import os
from openai import ChatCompletion


def answer_question(question: str) -> str:
    """Answer general questions using GPT-4."""
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
    )
    return response.choices[0].message["content"].strip()
