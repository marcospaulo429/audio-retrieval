from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity(audio_embedding, text_embedding):
    """
    Computes the cosine similarity between audio and text embeddings.

    Parameters:
    audio_embedding (np.ndarray): The embedding vector for the audio input.
    text_embedding (np.ndarray): The embedding vector for the text input.

    Returns:
    float: The cosine similarity score between the two embeddings.
    """
    audio_embedding = audio_embedding.reshape(1, -1)
    text_embedding = text_embedding.reshape(1, -1)
    similarity_score = cosine_similarity(audio_embedding, text_embedding)[0][0]
    return similarity_score

def retrieve_similar_items(query_embedding, embeddings_list):
    """
    Retrieves the most similar items from a list of embeddings based on cosine similarity.

    Parameters:
    query_embedding (np.ndarray): The embedding vector for the query input.
    embeddings_list (list of np.ndarray): A list of embedding vectors to compare against.

    Returns:
    list: Indices of the most similar items in the embeddings list.
    """
    similarities = [compute_similarity(query_embedding, emb) for emb in embeddings_list]
    similar_indices = np.argsort(similarities)[::-1]  # Sort in descending order
    return similar_indices