import unittest
from src.utils.embedding_utils import save_embedding, load_embedding
from src.utils.metrics import precision, recall

class TestUtils(unittest.TestCase):

    def test_save_load_embedding(self):
        embedding = [0.1, 0.2, 0.3]
        file_path = 'test_embedding.npy'
        
        save_embedding(embedding, file_path)
        loaded_embedding = load_embedding(file_path)
        
        self.assertEqual(embedding, loaded_embedding)

    def test_precision(self):
        true_positives = 10
        false_positives = 5
        expected_precision = true_positives / (true_positives + false_positives)
        
        self.assertAlmostEqual(precision(true_positives, false_positives), expected_precision)

    def test_recall(self):
        true_positives = 10
        false_negatives = 2
        expected_recall = true_positives / (true_positives + false_negatives)
        
        self.assertAlmostEqual(recall(true_positives, false_negatives), expected_recall)

if __name__ == '__main__':
    unittest.main()