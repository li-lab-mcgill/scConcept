# Baseline software versions and environments

This document records the principal software versions reported from the environments used for the ten baseline methods in the scConcept benchmark. 

## Baseline summary

| Baseline | Implementation | Version | Python or R | Environment |
|---|---|---|---|---|
| CellPLM | `cellplm` | 0.1.0.post2 | Python 3.9.25 | `cellplm` |
| scGPT | `scgpt` | 0.2.4 | Python 3.9.25 | `scgpt` |
| GenePT | Precomputed gene embeddings | See embedding source below | Python 3.11.14 | Same as scConcept |
| Cell2Sentence | `cell2sentence` | 1.2.0 | Python 3.8.20 | `cell2sentence` |
| scVI | `scvi-tools` | 1.3.3 | Python 3.10.19 | `scvi` |
| scVI-LD | `scvi-tools` | 1.3.3 | Python 3.10.19 | Same as scVI |
| scETM | `scETM` | 0.5.0 | Python 3.8.8 | `scetm38` |
| d-scIGM | GitHub source code | Commit not recorded | Python 3.8.8 | Same as scETM |
| scE2TM | GitHub source code | Commit not recorded | Python 3.8.8 | `scE2TM_env` |
| Seurat | R package `Seurat` | 4.4.0 | R 4.3.1 | `r-singlecell` |

## Key package versions

Only the principal runtime, model, data-processing, and evaluation packages are listed. Packages not installed in an environment are omitted. Python versions are runtime requirements and are not pip package entries.

### CellPLM

| Component | Version |
|---|---|
| Python | 3.9.25 |
| cellplm | 0.1.0.post2 |
| PyTorch | 1.13.0+cu117 |
| NumPy | 1.23.5 |
| SciPy | 1.13.1 |
| AnnData | 0.10.9 |
| Scanpy | 1.10.3 |
| scikit-learn | 1.6.1 |

### scGPT

| Component | Version |
|---|---|
| Python | 3.9.25 |
| scgpt | 0.2.4 |
| PyTorch | 2.0.1+cu118 |
| torchtext | 0.15.2 |
| NumPy | 1.24.4 |
| AnnData | 0.10.9 |
| Scanpy | 1.9.8 |
| scikit-learn | 1.6.1 |
| scIB | 1.1.7 |

### GenePT

GenePT used the same environment as scConcept. Precomputed gene embeddings were obtained from [Zenodo record 10833191](https://zenodo.org/records/10833191). Cell embeddings were computed by expression-weighted averaging of gene embeddings; the gene embedding model was not trained in this analysis.

| Component | Version |
|---|---|
| Python | 3.11.14 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| SciPy | 1.16.3 |
| AnnData | 0.12.6 |
| Scanpy | 1.11.5 |
| scikit-learn | 1.7.2 |

The exact embedding filename within the Zenodo record has not yet been recorded in this document. The archive identifier identifies the source, rather than a GenePT package version.

### Cell2Sentence

| Component | Version |
|---|---|
| Python | 3.8.20 |
| cell2sentence | 1.2.0 |
| PyTorch | 2.4.1 |
| Transformers | 4.46.3 |
| datasets | 3.1.0 |
| accelerate | 1.0.1 |
| NumPy | 1.24.4 |
| Scanpy | 1.9.8 |
| scikit-learn | 1.3.2 |

### scVI and scVI-LD

Both methods used the same `scvi-tools` installation and environment.

| Component | Version |
|---|---|
| Python | 3.10.19 |
| scvi-tools | 1.3.3 |
| PyTorch | 2.9.1 |
| lightning | 2.6.0 |
| pytorch-lightning | 2.6.0 |
| NumPy | 1.26.4 |
| AnnData | 0.11.4 |
| Scanpy | 1.11.5 |
| scikit-learn | 1.7.2 |

### scETM and d-scIGM

Both methods used the `scetm38` environment. scETM was available as an installed package. d-scIGM was run from [the authors' GitHub repository](https://github.com/nbnbhwyy/d-scIGM); no installed d-scIGM package version or source commit has been recorded.

| Component | Version |
|---|---|
| Python | 3.8.8 |
| scETM | 0.5.0 |
| PyTorch | 1.10.2+cu113 |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| AnnData | 0.9.2 |
| Scanpy | 1.9.1 |
| scikit-learn | 1.3.2 |

### scE2TM

scE2TM was run from [the authors' GitHub repository](https://github.com/nbnbhwyy/scE2TM). No installed scE2TM package version or source commit has been recorded.

| Component | Version |
|---|---|
| Python | 3.8.8 |
| PyTorch | 1.9.1+cu102 |
| NumPy | 1.24.4 |
| SciPy | 1.10.1 |
| AnnData | 0.9.2 |
| Scanpy | 1.9.8 |
| scikit-learn | 1.3.2 |
| POT | 0.8.2 |

### Seurat

| Component | Version |
|---|---|
| R | 4.3.1 |
| Seurat | 4.4.0 |
| SeuratObject | 5.2.0 |
| Matrix | 1.6-5 |
| dplyr | 1.2.1 |
| ggplot2 | 3.5.2 |
| igraph | 1.5.1 |
| uwot | 0.2.3 |

## Scope of this record

This document reports software versions and environment relationships. Hyperparameters not explicitly specified were set to default values as reported in the original publications or corresponding codebases.
