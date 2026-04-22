from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import io as sio
from scipy import sparse


class SingleCellDataset:
    """
    Load single-cell gene expression data from .mat files.

    Expected fields in .mat:
    - bow_train: (n_cells, n_genes)
    - bow_test:  (n_cells, n_genes)
    - label_train
    - label_test
    - voc: gene names
    """

    def __init__(self, dataset_name, batch_size, data_dir):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)

        (
            self.train_data,
            self.test_data,
            self.train_labels,
            self.test_labels,
            self.gene_names,
        ) = self.load_data()

        self.n_genes = len(self.gene_names)

        self._print_stats()

        # convert to torch (keep on CPU here)
        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)

        # dataloader
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False
        )

    def load_data(self):
        """
        Load .mat dataset.
        """
        file_path = self.data_dir / f"{self.dataset_name}.mat"

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        data_dict = sio.loadmat(file_path)

        train_data = sparse_to_dense(data_dict["bow_train"])
        test_data = sparse_to_dense(data_dict["bow_test"])

        # gene names
        gene_names = data_dict["voc"].reshape(-1).tolist()
        gene_names = [g[0] for g in gene_names]

        train_labels = data_dict["label_train"]
        test_labels = data_dict["label_test"]

        return train_data, test_data, train_labels, test_labels, gene_names

    def _print_stats(self):
        print("===> Train size:", self.train_data.shape[0])
        print("===> Test size:", self.test_data.shape[0])
        print("===> Number of genes:", self.n_genes)
        print("===> Avg expression per cell: {:.3f}".format(
            self.train_data.sum() / self.train_data.shape[0]
        ))
        print("===> Number of labels:", len(np.unique(self.train_labels)))


def sparse_to_dense(matrix):
    """
    Convert sparse matrix to dense float32 numpy array.
    """
    if sparse.isspmatrix(matrix):
        matrix = matrix.toarray()
    return matrix.astype("float32")