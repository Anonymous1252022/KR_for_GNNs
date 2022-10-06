import faiss
import numpy as np
from scipy import stats


class KNNFaiss:
    """
    Code is taken from:
    https://raw.githubusercontent.com/shankarpm/faiss_knn/master/faiss_knn.py
    """

    def __init__(self, k, gpu_idx: int = -1):
        self.k = k  # k nearest neighbor value
        self.index = 0
        self.gpu_idx = gpu_idx  # Which GPU index to use
        self.train_labels = []
        self.test_label_faiss_output = []

    def fitModel(self, train_features, train_labels):
        self.gpu_index_flat = self.index = faiss.IndexFlatL2(train_features.shape[1])  # build the index
        try:
            res = faiss.StandardGpuResources()
            self.gpu_index_flat = faiss.index_cpu_to_gpu(res, self.gpu_idx, self.gpu_index_flat)
        except AttributeError:
            res = None
        self.train_labels = train_labels
        self.gpu_index_flat.add(train_features)  # add vectors to the index

    def predict(self, test_features):
        distance, test_features_faiss_Index = self.gpu_index_flat.search(test_features, self.k)
        self.test_label_faiss_output = stats.mode(self.train_labels[test_features_faiss_Index], axis=1)[0]
        self.test_label_faiss_output = np.array(self.test_label_faiss_output.ravel())
        return self.test_label_faiss_output
