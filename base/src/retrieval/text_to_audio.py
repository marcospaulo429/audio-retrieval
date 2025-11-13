def retrieve_audio_from_text(text_query, audio_embeddings, text_embeddings, joint_embedding_model):
    """
    Retrieve the most similar audio file based on a text query.

    Parameters:
    - text_query: The input text query for which to find the corresponding audio.
    - audio_embeddings: Precomputed audio embeddings.
    - text_embeddings: Precomputed text embeddings.
    - joint_embedding_model: The model used to compute joint embeddings.

    Returns:
    - The audio file corresponding to the most similar text query.
    """
    # Encode the text query into an embedding
    text_embedding = joint_embedding_model.encode_text(text_query)

    # Compute similarities between the text embedding and audio embeddings
    similarities = compute_similarity(text_embedding, audio_embeddings)

    # Retrieve the index of the most similar audio
    most_similar_index = similarities.argmax()

    # Return the corresponding audio file
    return audio_embeddings[most_similar_index]


def compute_similarity(query_embedding, target_embeddings):
    """
    Compute similarity scores between a query embedding and a set of target embeddings.

    Parameters:
    - query_embedding: The embedding of the query (text or audio).
    - target_embeddings: The embeddings of the target items (audio or text).

    Returns:
    - An array of similarity scores.
    """
    # Implement cosine similarity or any other similarity measure
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(query_embedding.reshape(1, -1), target_embeddings).flatten()