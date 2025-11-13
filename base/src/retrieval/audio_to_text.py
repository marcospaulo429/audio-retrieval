from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.joint_embedding import JointEmbedding
from utils.embedding_utils import load_embedding
from utils.metrics import compute_similarity

def retrieve_text_from_audio(audio_query, audio_encoder, text_encoder, joint_embedding, threshold=0.5):
    audio_embedding = audio_encoder.encode_audio(audio_query)
    text_embeddings = load_embedding('text_embeddings')  # Assuming text embeddings are precomputed and stored
    similarities = compute_similarity(audio_embedding, text_embeddings)

    # Retrieve the most similar text based on the threshold
    retrieved_texts = []
    for idx, similarity in enumerate(similarities):
        if similarity >= threshold:
            retrieved_texts.append((idx, similarity))

    # Sort by similarity score
    retrieved_texts.sort(key=lambda x: x[1], reverse=True)
    return retrieved_texts

# Example usage
if __name__ == "__main__":
    audio_encoder = AudioEncoder()
    text_encoder = TextEncoder()
    joint_embedding = JointEmbedding()

    # Load models
    audio_encoder.load_model()
    text_encoder.load_model()

    # Example audio query
    audio_query = "path/to/audio/file.wav"
    results = retrieve_text_from_audio(audio_query, audio_encoder, text_encoder, joint_embedding)
    print("Retrieved Texts:", results)