from rag.ingestion import build_index

import warnings
warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

build_index("test.pdf", "chroma_index")
