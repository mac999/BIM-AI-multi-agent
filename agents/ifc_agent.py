import json
from pathlib import Path

import ifcopenshell
from openai import ChatCompletion


def parse_ifc_to_json(ifc_path: str) -> str:
    model = ifcopenshell.open(ifc_path)
    data = {"entities": len(model)}
    json_path = Path(ifc_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return json_path.as_posix()


def answer_question(ifc_path: str, question: str) -> str:
    json_path = parse_ifc_to_json(ifc_path)
    with open(json_path, "r", encoding="utf-8") as f:
        info = f.read()
    prompt = f"IFC 정보:\n{info}\n\n질문: {question}"
    response = ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "IFC 파일 내용을 바탕으로 답변."},
                 {"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message["content"].strip()
