import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import joblib


df=pd.read_csv("movies.csv") 

model=SentenceTransformer('paraphase-MiniLM-L6-v2')

try:

    embeddings=joblib.load("cached_embeddings.pkl")
    print("Loaded cached embeddings.")

except FileNotFoundError:

    print("Generating embeddings. This may take a few minutes...")

    embeddings=model.encode(df['storyline'].tolist(). show_progress_bar-True, batch_size =64) 

joblib.dump(embeddings, "cached_embeddings.pkl")
print("Embeddings saved to cache.")


def recommend_movies(input_storyline, top_n=5):


    input_embedding=model.encode([input_storyline])

    similarities=cosine_similarity(input_embedding, embeddings)





    top_indices=similarities[0].argsort()[-top_n:] [::-1] 