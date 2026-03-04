from rag.pipeline import RAGPipeline

import warnings
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

rag = RAGPipeline(persist_dir="chroma_index")
result = rag.answer_question("Does this paper mention dermatophytes?")
print(result["answer"])
print(result["sources"][0])