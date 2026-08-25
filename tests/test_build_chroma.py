import pandas as pd
import chromadb
from src.shared.paths import CHROMA_COLLECTION, CHROMA_DIR, PREPROCESSED_PARQUET


def test_chroma_length_matches_df():
    df = pd.read_parquet(PREPROCESSED_PARQUET)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    assert collection.count() == len(df)
