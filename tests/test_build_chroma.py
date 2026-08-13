import pandas as pd
import chromadb
from src.retrieval_pipeline.build_chroma import PARQUET_PATH, CHROMA_PATH, COLLECTION_NAME


def test_chroma_length_matches_df():
    df = pd.read_parquet(PARQUET_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)
    assert collection.count() == len(df)
