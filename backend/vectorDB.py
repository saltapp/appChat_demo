import chromadb
from chromadb import Settings

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from config import CHROMA_DATA_PATH,CHROMA_TENANT, CHROMA_DATABASE

CHROMA_CLIENT = chromadb.PersistentClient(
        path=CHROMA_DATA_PATH,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

# Create LangChain Chroma instance
langchain_chroma = Chroma(
    client=CHROMA_CLIENT,
    embedding_function=OpenAIEmbeddings(),  # Replace with your preferred embedding function
)