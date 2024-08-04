import chromadb
from chromadb import Settings
from config import CHROMA_DATA_PATH,CHROMA_TENANT, CHROMA_DATABASE


CHROMA_CLIENT = chromadb.PersistentClient(
        path=CHROMA_DATA_PATH,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )