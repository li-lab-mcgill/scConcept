import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from Runner import Runner
from singlecell_dataset import SingleCellDataset


def parse_args():
    """
    Parse command-line arguments for ECRTM training.
    """
    parser = argparse.ArgumentParser(
        description="Train ECRTM on single-cell gene expression data."
    )

    # Basic settings
    parser.add_argument("--dataset", type=str, default="pollen",
                        help="Dataset name (without .mat suffix)")
    parser.add_argument("--n_topic", type=int, default=50,
                        help="Number of topics")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")

    # Training
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr_scheduler", type=bool, default=True,
                        help="Whether to use learning rate scheduler")
    parser.add_argument("--lr_step_size", type=int, default=25,
                        help="Step size for learning rate scheduler")

    # ECR
    parser.add_argument("--sinkhorn_alpha", type=float, default=20,
                        help="Sinkhorn regularization strength")
    parser.add_argument("--OT_max_iter", type=int, default=1000,
                        help="Max Sinkhorn iterations")
    parser.add_argument("--weight_loss_ECR", type=float, default=100,
                        help="Weight of ECR loss")
    parser.add_argument("--eval_step", type=int, default=50,
                        help="Save topics and model every N epochs")

    # Model
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--en1_units", type=int, default=200)
    parser.add_argument("--beta_temp", type=float, default=0.2)

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cpu", "cuda", "cuda:0", "cuda:1"'
    )

    # Path
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing .mat datasets")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for saving training outputs.")

    args = parser.parse_args()

    # resolve default dataset path
    if args.data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        args.data_dir = str(project_root / "Datasets")

    if args.output_dir is None:
        ecrtm_root = Path(__file__).resolve().parent
        args.output_dir = str(ecrtm_root / "output")

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"  
        else:
            args.device = "cpu"

    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 70)
    print("=" * 70)
    print("\n" + yaml.dump(vars(args), default_flow_style=False))

    # load dataset
    dataset = SingleCellDataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
    )

    # attach dataset info
    args.vocab_size = dataset.n_genes

    # train
    runner = Runner(args, dataset)
    runner.train(dataset.train_loader)


if __name__ == "__main__":
    main()