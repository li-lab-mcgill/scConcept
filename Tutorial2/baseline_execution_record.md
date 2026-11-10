# Baseline execution record

This record documents launch commands and notebook entry points for selected scConcept benchmark baselines. See [Baseline software versions and environments](baseline_software_environments.md) for the corresponding environment records. The standard benchmark comparison used Louvain resolution 1.

## d-scIGM

The benchmark batch script was `launch_all_gbn.sh` in the d-scIGM working directory. The saved Pollen log records the following invocation, with the original server-specific path shortened to a path relative to that working directory.

```bash
conda activate scetm38
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python model/main.py --dataname pollen_HIGHPRE_10000.csv --labelname pollen_cell_anno.csv
```

Run this command from the d-scIGM source directory with the input layout expected by its entry point. For other datasets, substitute the corresponding `<dataset>_HIGHPRE_10000.csv` expression file and `<dataset>_cell_anno.csv` annotation file. The batch script invoked the same entry point sequentially for the configured datasets. Its master log records successful completion of all 17 entries in that batch between April 6 and 9, 2026. The working directory has no Git metadata from which to recover the source commit.

The `d-scIGM` section of `cluster_performance_fin.ipynb` evaluates the saved embeddings using their first 50 columns, a Scanpy neighbor graph, and Louvain clustering with resolution 1 and seed 1. It also exports the top 100 genes per topic from the saved topic–gene matrices.

## Notebook entry points

The original evaluation notebook was `cluster_performance_fin.ipynb`. The table identifies the section to run and the inputs it expects. Input and output paths in the original notebook are specific to the compute server.

| Baseline | Notebook section | Recorded execution or inputs |
| --- | --- | --- |
| scVI | `scVI` | The 50-dimensional benchmark cell calls `scvi.model.SCVI.setup_anndata`, trains `scvi.model.SCVI(n_latent=50)` with `model.train(devices=1)`, and extracts the latent representation with `get_latent_representation()`. For the standard comparison, set Louvain resolution to 1 and use seed 1. |
| scVI-LD | `scVI-ld` | Calls `scvi.model.LinearSCVI.setup_anndata`, trains `scvi.model.LinearSCVI(n_latent=50)` with `model.train(devices=1)`, and evaluates the latent embedding with Louvain resolution 1 and seed 1. It uses `get_loadings()` to export the top 100 genes per latent dimension. |
| scETM | `scETM` | Trains the 50-dimensional scETM model with `trainer.train(..., n_epochs=800)`. Evaluates the saved latent representation with Louvain resolution 1 and seed 1 and exports the top 100 genes per topic. |
| GenePT | `GenePT` | Loads `GenePT_gene_embedding_ada_text.pickle` from [Zenodo record 10833191](https://zenodo.org/records/10833191). Computes expression-weighted cell embeddings from matched genes and applies 50-component PCA, a Scanpy neighbor graph, and Louvain clustering with resolution 1 and seed 1. |
| Cell2Sentence | `cts` | Reads precomputed `<dataset>_cell_embedding.npy` files and applies 50-component PCA, a Scanpy neighbor graph, and Louvain clustering with resolution 1 and seed 1. |
| CellPLM | `cellplm` | Reads precomputed `<dataset>_cell_embedding.csv` files and applies 50-component PCA, a Scanpy neighbor graph, and Louvain clustering with resolution 1 and seed 1. |
| scGPT | `scGPT` | Reads precomputed `<dataset>_scGPT.csv` files and applies 50-component PCA, a Scanpy neighbor graph, and Louvain clustering with resolution 1 and seed 1. |
| scE2TM | `scE2TM` | Reads the saved `scE2TM<dataset>_embedding.csv` and `scE2TM<dataset>_w_gene_E_tg.csv` files. Evaluates the embeddings with a Scanpy neighbor graph and Louvain clustering with resolution 1 and seed 1 and exports the top 100 genes per topic. |

