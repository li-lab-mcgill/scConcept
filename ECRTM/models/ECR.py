import torch
from torch import nn

class ECR(nn.Module):
    """
    Embedding Clustering Regularization (ECR) module.

    This module computes an optimal transport-based regularization loss
    between topic embeddings and gene embeddings using the Sinkhorn algorithm.

    Parameters
    ----------
    weight_loss_ECR : float
        Weight applied to the final ECR loss.
    sinkhorn_alpha : float
        Entropic regularization strength used in the Sinkhorn kernel.
    OT_max_iter : int, default=1000
        Maximum number of Sinkhorn iterations.
    stopThr : float, default=0.5e-2
        Stopping threshold for Sinkhorn iterations.
    """

    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=1000, stopThr=0.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, cost_matrix):
        """
        Compute the ECR loss from a topic-gene cost matrix.

        Parameters
        ----------
        cost_matrix : torch.Tensor
            Pairwise cost matrix of shape (n_topics, n_genes).

        Returns
        -------
        torch.Tensor
            Scalar ECR loss.
        """
        device = cost_matrix.device

        n_topics, n_genes = cost_matrix.shape

        # Uniform marginals for topics and genes
        topic_marginal = (torch.ones(n_topics, device=device) / n_topics).unsqueeze(1)
        gene_marginal = (torch.ones(n_genes, device=device) / n_genes).unsqueeze(1)

        # Initialize Sinkhorn scaling vector
        u = torch.ones_like(topic_marginal, device=device) / n_topics

        # Sinkhorn kernel
        kernel = torch.exp(-cost_matrix * self.sinkhorn_alpha)

        error = 1.0
        iteration = 0

        while error > self.stopThr and iteration < self.OT_max_iter:
            v = gene_marginal / (torch.matmul(kernel.t(), u) + self.epsilon)
            u = topic_marginal / (torch.matmul(kernel, v) + self.epsilon)
            iteration += 1

            if iteration % 50 == 1:
                transported_gene_marginal = v * torch.matmul(kernel.t(), u)
                error = torch.norm(
                    torch.sum(torch.abs(transported_gene_marginal - gene_marginal), dim=0),
                    p=float("inf")
                )

        transport_plan = u * (kernel * v.T)

        loss_ecr = torch.sum(transport_plan * cost_matrix)
        loss_ecr = loss_ecr * self.weight_loss_ECR

        return loss_ecr