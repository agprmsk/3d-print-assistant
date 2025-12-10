from .agents import classify_query, retrieve_knowledge, consultant_answer, safety_check

def handle_user_query(user_query: str, dialog_context: str = "") -> dict:
    category = classify_query(user_query)
    docs = retrieve_knowledge(user_query, k=6)
    raw_answer = consultant_answer(user_query, docs, dialog_context=dialog_context)
    safe_answer = safety_check(raw_answer)

    sources = list({d.metadata.get("source_url") for d in docs})
    return {
        "category": category,
        "answer": safe_answer,
        "sources": sources,
    }
