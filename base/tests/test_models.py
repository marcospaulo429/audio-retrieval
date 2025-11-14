import unittest
from src.models.audio_encoder import AudioEncoder
from src.models.text_encoder import TextEncoder
from src.models.joint_embedding import JointEmbedding

class TestAudioTextRetrievalModels(unittest.TestCase):

    def setUp(self):
        self.audio_encoder = AudioEncoder()
        self.text_encoder = TextEncoder()
        self.joint_embedding = JointEmbedding()

    def test_audio_encoder(self):
        audio_input = "path/to/audio/file.wav"
        embedding = self.audio_encoder.encode_audio(audio_input)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), self.audio_encoder.embedding_dim)

    def test_text_encoder(self):
        text_input = "This is a sample text."
        embedding = self.text_encoder.encode_text(text_input)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), self.text_encoder.embedding_dim)

    def test_joint_embedding(self):
        audio_embedding = self.audio_encoder.encode_audio("path/to/audio/file.wav")
        text_embedding = self.text_encoder.encode_text("This is a sample text.")
        joint_embedding_result = self.joint_embedding.compute_joint_embedding(audio_embedding, text_embedding)
        self.assertIsNotNone(joint_embedding_result)

if __name__ == '__main__':
    unittest.main()