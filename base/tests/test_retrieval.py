import unittest
from src.retrieval.audio_to_text import retrieve_text_from_audio
from src.retrieval.text_to_audio import retrieve_audio_from_text
from src.retrieval.similarity import compute_similarity

class TestRetrievalFunctions(unittest.TestCase):

    def setUp(self):
        self.audio_sample = "path/to/sample_audio.wav"
        self.text_sample = "This is a sample text description."
        self.expected_text = "Expected text description for the audio."
        self.expected_audio = "path/to/expected_audio.wav"

    def test_retrieve_text_from_audio(self):
        retrieved_text = retrieve_text_from_audio(self.audio_sample)
        self.assertEqual(retrieved_text, self.expected_text)

    def test_retrieve_audio_from_text(self):
        retrieved_audio = retrieve_audio_from_text(self.text_sample)
        self.assertEqual(retrieved_audio, self.expected_audio)

    def test_compute_similarity(self):
        audio_embedding = [0.1, 0.2, 0.3]
        text_embedding = [0.1, 0.2, 0.3]
        similarity_score = compute_similarity(audio_embedding, text_embedding)
        self.assertGreaterEqual(similarity_score, 0)

if __name__ == '__main__':
    unittest.main()