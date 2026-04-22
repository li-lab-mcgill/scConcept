import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
try:
    from .ECR import ECR
except ImportError:
    from ECR import ECR


class ECRTM(nn.Module):
    """
    ECRTM model for single-cell gene expression analysis.

    This model learns:
    - a topic proportion for each cell
    - a topic-gene distribution for each topic
    - a regularized embedding geometry via ECR

    Parameters
    ----------
    args : argparse.Namespace
        Configuration object containing model and training hyperparameters.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.beta_temp = args.beta_temp

        # Logistic normal prior parameters for topic proportions
        prior_alpha = np.ones((1, args.n_topic), dtype=np.float32)
        prior_mean = (np.log(prior_alpha).T - np.mean(np.log(prior_alpha), axis=1)).T
        prior_var = (
            ((1.0 / prior_alpha) * (1 - (2.0 / args.n_topic))).T
            + (1.0 / (args.n_topic * args.n_topic)) * np.sum(1.0 / prior_alpha, axis=1)
        ).T

        self.prior_mean = nn.Parameter(torch.as_tensor(prior_mean), requires_grad=False)
        self.prior_var = nn.Parameter(torch.as_tensor(prior_var), requires_grad=False)

        # Encoder layers for inferring cell-topic proportions
        self.fc_mean = nn.Linear(args.en1_units, args.n_topic)
        self.fc_logvar = nn.Linear(args.en1_units, args.n_topic)
        self.encoder_dropout = nn.Dropout(args.dropout)

        self.mean_bn = nn.BatchNorm1d(args.n_topic)
        self.logvar_bn = nn.BatchNorm1d(args.n_topic)
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)

        # Gene embeddings
        self.gene_embeddings = torch.empty((args.vocab_size, args.en1_units))
        nn.init.trunc_normal_(self.gene_embeddings, std=0.02)
        self.gene_embeddings = nn.Parameter(F.normalize(self.gene_embeddings))

        # Topic embeddings
        self.topic_embeddings = torch.empty((args.n_topic, args.en1_units))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.02)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        # Embedding clustering regularization module
        self.ecr = ECR(
            weight_loss_ECR=args.weight_loss_ECR,
            sinkhorn_alpha=args.sinkhorn_alpha,
            OT_max_iter=args.OT_max_iter,
        )

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(args.vocab_size, args.en1_units),
            nn.BatchNorm1d(args.en1_units),
            nn.Tanh(),
            nn.Linear(args.en1_units, args.en1_units),
        )

    def get_beta(self):
        """
        Compute the topic-gene distribution matrix.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_topics, n_genes), where each row corresponds
            to the gene distribution of a topic.
        """
        distance_matrix = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.gene_embeddings
        )
        beta = F.softmax(-distance_matrix / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick for sampling latent topic logits.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            return mean + noise * std
        return mean

    def encode(self, cell_expression):
        """
        Encode cell-by-gene expression into cell-topic proportions.

        Parameters
        ----------
        cell_expression : torch.Tensor
            Input expression matrix of shape (batch_size, n_genes).

        Returns
        -------
        tuple
            theta : torch.Tensor
                Cell-topic proportions of shape (batch_size, n_topics).
            kl_loss : torch.Tensor
                KL divergence loss term.
        """
        hidden = self.encoder(cell_expression)
        hidden = self.encoder_dropout(hidden)

        mean = self.mean_bn(self.fc_mean(hidden))
        logvar = self.logvar_bn(self.fc_logvar(hidden))

        latent = self.reparameterize(mean, logvar)
        theta = F.softmax(latent, dim=1)

        kl_loss = self.compute_kl_loss(mean, logvar)

        return theta, kl_loss

    def get_theta(self, cell_expression):
        """
        Return cell-topic proportions.

        During training, also returns the KL loss.
        During evaluation, returns only theta.
        """
        theta, kl_loss = self.encode(cell_expression)
        if self.training:
            return theta, kl_loss
        return theta

    def compute_kl_loss(self, mean, logvar):
        """
        Compute the KL divergence between the inferred posterior and prior.
        """
        var = logvar.exp()
        var_ratio = var / self.prior_var
        mean_diff = mean - self.prior_mean
        mean_diff_term = mean_diff * mean_diff / self.prior_var
        logvar_diff = self.prior_var.log() - logvar

        kl_divergence = 0.5 * (
            (var_ratio + mean_diff_term + logvar_diff).sum(axis=1) - self.args.n_topic
        )
        kl_divergence = kl_divergence.mean()
        return kl_divergence

    def get_ecr_loss(self):
        """
        Compute the embedding clustering regularization loss.
        """
        cost_matrix = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.gene_embeddings
        )
        ecr_loss = self.ecr(cost_matrix)
        return ecr_loss

    @staticmethod
    def pairwise_euclidean_distance(x, y):
        """
        Compute pairwise squared Euclidean distance between two matrices.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (n_x, d).
        y : torch.Tensor
            Tensor of shape (n_y, d).

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix of shape (n_x, n_y).
        """
        return (
            torch.sum(x ** 2, dim=1, keepdim=True)
            + torch.sum(y ** 2, dim=1)
            - 2 * torch.matmul(x, y.t())
        )

    def forward(self, cell_expression, epoch=None):
        """
        Forward pass for one mini-batch.

        Parameters
        ----------
        cell_expression : torch.Tensor
            Input expression matrix of shape (batch_size, n_genes).
        epoch : int, optional
            Training epoch. Currently unused, kept for compatibility.

        Returns
        -------
        dict
            Dictionary containing total loss and individual loss terms.
        """
        theta, kl_loss = self.encode(cell_expression)
        beta = self.get_beta()

        reconstructed_expression = F.softmax(
            self.decoder_bn(torch.matmul(theta, beta)),
            dim=-1
        )
        reconstruction_loss = -(cell_expression * reconstructed_expression.log()).sum(axis=1).mean()

        topic_model_loss = reconstruction_loss + kl_loss
        ecr_loss = self.get_ecr_loss()
        total_loss = topic_model_loss + ecr_loss

        loss_dict = {
            "loss": total_loss,
            "loss_TM": topic_model_loss,
            "loss_ECR": ecr_loss,
        }

        return loss_dict