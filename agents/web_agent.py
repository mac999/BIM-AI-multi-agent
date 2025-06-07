import os
from typing import List

import requests
from openai import ChatCompletion

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def _search_web(query: str) -> List[str]:
    if SERPER_API_KEY:
        resp = requests.post(
            "https://google.serper.dev/search",
            json={"q": query},
            headers={"X-API-KEY": SERPER_API_KEY},
            timeout=10,
        )
        data = resp.json()
        return [item["snippet"] for item in data.get("organic", [])]
    elif TAVILY_API_KEY:
        resp = requests.get(
            "https://api.tavily.com/search",
            params={"api_key": TAVILY_API_KEY, "query": query},
            timeout=10,
        )
        data = resp.json()
        return [d["content"] for d in data.get("results", [])]
    else:
        return []


def search_and_summarize(query: str) -> str:
    snippets = _search_web(query)
    joined = "\n".join(snippets)
    prompt = f"""다음 웹 검색 결과를 기반으로 질문에 한국어로 답하세요:
{joined}
질문: {query}
"""
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "웹 검색 결과를 요약하여 답변."},
                 {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message["content"].strip()
