from app.embedder import embed_texts, semantic_search
from app.llm import ask_llm

def answer_questions(full_text, questions, meta=None):
    """
    Given the full text of a document, answer a list of questions using embeddings + Together AI LLM.
    """
    chunks = full_text.split("\n\n")
    embeddings = embed_texts(chunks)

    answers = []
    for q in questions:
        top_chunks = semantic_search(q, chunks, embeddings, top_k=6)
        context = "\n\n".join(top_chunks)
        ans = ask_llm(q, context)
        answers.append(ans)

    return answers
