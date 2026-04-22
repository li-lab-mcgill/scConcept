import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

try:
    from .models.ECRTM import ECRTM
except ImportError:
    from models.ECRTM import ECRTM


class Runner:
    """
    Training and evaluation wrapper for ECRTM.
    """

    def __init__(self, args, dataset_handler):
        self.args = args
        self.dataset_handler = dataset_handler

        # device
        self.device = torch.device(args.device)

        self.model = ECRTM(args).to(self.device)

        # Gene names from the dataset
        self.gene_names = dataset_handler.gene_names
        self.gene_to_idx = {gene: idx for idx, gene in enumerate(self.gene_names)}

        print(f"Number of topics: {args.n_topic}")
        print(f"Using device: {self.device}")

        self.run_name = f"{args.dataset}_K{args.n_topic}_seed{args.seed}"

    def make_optimizer(self):
        """
        Create optimizer.
        """
        return torch.optim.RMSprop(
            params=self.model.parameters(),
            lr=self.args.lr,
        )

    def make_lr_scheduler(self, optimizer):
        """
        Create learning rate scheduler.
        """
        return StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5)

    def train(self, data_loader):
        """
        Train the ECRTM model.

        Parameters
        ----------
        data_loader : DataLoader
            Training data loader.

        Returns
        -------
        np.ndarray
            Final topic-gene distribution matrix.
        """
        optimizer = self.make_optimizer()
        lr_scheduler = None

        if self.args.lr_scheduler:
            print("===> Using learning rate scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(data_loader.dataset)

        for epoch in range(self.args.epochs):
            self.model.train()
            loss_sums = defaultdict(float)

            for batch_data in data_loader:
                batch_data = batch_data.to(self.device)

                loss_dict = self.model(batch_data, epoch)
                total_loss = loss_dict["loss"]

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_size = len(batch_data)
                for key, value in loss_dict.items():
                    loss_sums[key] += value.item() * batch_size

            if lr_scheduler is not None:
                lr_scheduler.step()

            log_message = f"Epoch: {epoch:03d}"
            for key, value in loss_sums.items():
                log_message += f" {key}: {value / data_size:.3f}"
            print(log_message)

            # Save topics and model checkpoints periodically
            if (epoch + 1) % self.args.eval_step == 0:
                self.model.eval()

                output_root = Path(self.args.output_dir)
                topic_dir = output_root / "topics" / self.run_name
                topic_dir.mkdir(parents=True, exist_ok=True)

                beta = self.model.get_beta().detach().cpu().numpy()
                topic_gene_lists = self.get_top_genes_per_topic(
                    beta=beta,
                    gene_names=self.gene_names,
                    num_top_genes=100,
                )

                topic_file = os.path.join(topic_dir, f"epoch{epoch + 1}_top_genes.txt")
                with open(topic_file, "w", encoding="utf-8") as file:
                    for gene_list in topic_gene_lists:
                        file.write(" ".join(gene_list) + "\n")

                checkpoint_dir = output_root / "models" / self.run_name
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = os.path.join(checkpoint_dir, f"epoch-{epoch + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

                self.model.train()

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def test(self, input_data):
        """
        Infer cell-topic proportions for input cells.

        Parameters
        ----------
        input_data : torch.Tensor
            Cell-by-gene expression matrix.

        Returns
        -------
        np.ndarray
            Cell-topic proportion matrix.
        """
        data_size = input_data.shape[0]
        theta = np.zeros((data_size, self.args.n_topic))
        all_indices = torch.split(torch.arange(data_size), self.args.batch_size)

        with torch.no_grad():
            self.model.eval()
            for indices in all_indices:
                batch_input = input_data[indices].to(self.device)
                batch_theta = self.model.get_theta(batch_input)

                if isinstance(batch_theta, tuple):
                    batch_theta = batch_theta[0]

                theta[indices] = batch_theta.detach().cpu().numpy()

        return theta

    @staticmethod
    def get_top_genes_per_topic(beta, gene_names, num_top_genes):
        """
        Get top genes for each topic.
        """
        topic_gene_lists = []
        gene_names = np.array(gene_names)

        for topic_distribution in beta:
            top_gene_indices = np.argsort(topic_distribution)[::-1][:num_top_genes]
            top_genes = gene_names[top_gene_indices].tolist()
            topic_gene_lists.append(top_genes)

        return topic_gene_lists