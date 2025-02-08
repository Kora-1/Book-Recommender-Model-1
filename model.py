from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import pandas as pd

# Global variables
embeddings, similarity_matrix, model, df = None, None, None, None


# Function to load necessary files and model
def load():
    global embeddings, similarity_matrix, model, df

    # Check if already loaded, if not, then load
    if embeddings is None:
        embeddings = pickle.load(open("embeddings.pkl", 'rb'))

    if similarity_matrix is None:
        similarity_matrix = pickle.load(
            open("similarity_matrix.pkl", 'rb'))

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    if df is None:
        df = pd.read_csv("Cleaned_Book_dataset2.csv")


# Function to recommend books based on user query
def recommend_books(query, top_n=5):
    # Ensure the model and other resources are loaded
    if model is None or embeddings is None or df is None:
        load()

    # Convert user input (query) to an embedding
    query_embedding = model.encode([query])

    # Compute similarity between the query and all book embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get the indices of top N most similar books
    similar_books_indices = np.argsort(similarities)[::-1][:top_n]

    # Return the book names and authors
    return df.iloc[similar_books_indices][['Book', 'Author']]

# Example Usage:
# print(recommend_books("J.K. Rowling", top_n=5))
# print(recommend_books("Fantasy", top_n=5))
# print(recommend_books("The Hobbit", top_n=5))
