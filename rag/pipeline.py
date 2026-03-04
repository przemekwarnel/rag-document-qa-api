import re
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Any


def get_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)

class STEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist() # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

def get_embeddings():
    return STEmbeddings()

def load_vectordb(persist_dir: str):
    embeddings = get_embeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )


class RAGPipeline:
    def __init__(self, persist_dir: str, k: int = 4, max_context_chars: int = 1500):
        self.llm = get_llm()
        self.vectordb = load_vectordb(persist_dir)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": k})
        self.max_context_chars = max_context_chars
    
    def _is_mention_query(self, query: str) -> bool:
        q = query.strip().lower()
        return any(p in q for p in ["mention", "contain", "include", "occurr", "appear"])
    
    def _extract_term(self, query: str) -> Optional[str]:
        q = query.strip()
        # match: mention X / contain X / include X
        m = re.search(r"(?:mention|contain|include|occur(?:ence)?|appear(?:s)?)\s+(.*?)[\?\.]?$", q, flags=re.IGNORECASE)
        if not m:
            return None
        term = m.group(1).strip().strip('"').strip("'")
        term = re.sub(r"^(the\s+term\s+)", "", term, flags=re.IGNORECASE).strip()
        if len(term) < 3:
            return None
        return term

    def answer_question(self, query: str) -> dict[str, Any]:
        docs = self.retriever.invoke(query)

        # Add 3 sources to the answer for better traceability
        sources = []
        for d in docs[:3]:
            snippet = d.page_content[:300].replace("\n", " ").strip()
            sources.append(
                {
                    "snippet": snippet,
                    "metadata": d.metadata
                }
            )

        # lexical presence check for "mention/contain/include" questions
        if self._is_mention_query(query):
            term = self._extract_term(query)
            if term:
                term_l = term.lower()

                hits = []
                for d in docs:
                    if term_l in d.page_content.lower():
                        hits.append(d)

                if hits:
                    hit_sources = []
                    for d in hits[:3]:
                        hit_sources.append({
                            "snippet": d.page_content[:300].replace("\n", " ").strip(),
                            "metadata": d.metadata
                        })
                    return {
                        "answer": "yes",
                        "sources": hit_sources,
                    }
                return {
                    "answer": "not found in top-k retrieved chunks",
                    "sources": sources,
                }
    
        context = ""
        for d in docs:
            if len(context) + len(d.page_content) > self.max_context_chars:
                break
            context += d.page_content + "\n\n"

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:"
        )
        response: Any = self.llm(prompt)
        result = response[0]["generated_text"] 
        return {"answer": result, "sources": sources}