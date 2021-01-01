import unittest
import torch
from sklearn.metrics import jaccard_score
import numpy as np

from code.search.model_custom_metrics.mIoU import calculate_IoU_per_batch

class TestmIoU(unittest.TestCase):

    def test_miou_basic(self):
        # B, C, H, W
        batch_size = np.random.randint(10)
        logits = np.random.random((batch_size, 3, 10, 20))
        labels = np.random.randint(0, 3, size=(batch_size, 10, 20))
        miou = calculate_IoU_per_batch(torch.from_numpy(logits), torch.from_numpy(labels), verbose=True).numpy()
        y_true = np.reshape(labels, [batch_size, -1])
        y_pred = np.reshape(np.argmax(logits, axis=1), [batch_size, -1])
        mean_iou = np.mean([jaccard_score(y_true[i], y_pred[i], average='macro') for i in range(batch_size)])
        self.assertEqual(miou, mean_iou)

    def test_miou_basic_all_correct(self):
        # B, C, H, W
        logits = np.random.random((5, 10, 10, 20))
        labels = np.argmax(logits, axis=1)
        iou = calculate_IoU_per_batch(torch.from_numpy(logits), torch.from_numpy(labels), verbose=True).numpy()
        self.assertEqual(iou, 1.)


if __name__ == '__main__':
    unittest.main()
