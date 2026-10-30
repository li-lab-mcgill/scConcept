# Downstream analysis tutorials

These notebooks start after scConcept installation, data preprocessing, and concept generation described in `Tutorial/tutorial_0_colab_quickstart.ipynb` and Tutorials 1–3. Use the deposited concepts to reproduce the reference analyses, or set the concept paths to your saved scConcept outputs.

| Notebook in `Tutorial2/` | Analysis |
|---|---|
| `tutorial_figure3_melanoma_analysis.ipynb` | Melanoma annotation, perturbation, SHAP/GSEA, and survival |
| `tutorial_figure4_hierarchical_analysis.ipynb` | Pancreas and lung hierarchical annotation |
| `tutorial_figure5_developmental_potency.ipynb` | Developmental potency and bidirectional perturbation |
| `tutorial_figure6_lung_cancer_analysis.ipynb` | Lung atlas, clinical associations, SCISSOR, and drug ranking |
| `tutorial_cross_dataset_transfer.ipynb` | Six transfer directions and cell-composition shifts |
| `tutorial_concept_robustness.ipynb` | Repeated calls, prompt sensitivity, and GPT/Claude agreement |

Open notebooks from the repository or any of its subdirectories, then run cells in order. The setup cell locates `scConcept.py`, imports the repository API, and configures inputs. No installation, topic training, or concept generation is repeated. Optional Claude rescoring requires a separate API key; it is disabled by default.

## Inputs

Processed `.mat` datasets default to `scConcept/Datasets/`. Their fields follow `ECRTM.singlecell_dataset.SingleCellDataset`: `bow_train`, `bow_test`, `label_train`, `label_test`, and `voc`. Analyses that only read the test split can retain sparse matrices without loading the training split. Large-atlas and clinical analyses also require the H5AD, CSV, or TSV files listed in their setup cells.

To use a different input directory, set `SCCONCEPT_DATA_DIR` before running the setup cell. For individual paths, copy `Inputs/data_paths.example.json` to `Inputs/data_paths.local.json` and edit the values. Relative values resolve under the data directory; absolute values are supported. Alternatively, set `SCCONCEPT_INPUT_CONFIG` to another JSON path map. Local path maps are excluded from version control.

The six notebooks are directly in `Tutorial2/`. Saved concepts remain in `Concept/`. Required scores, drug-analysis inputs, prompts, and reference comparisons are supplied separately in `reference_outputs.zip`; extract it into the repository root to populate `Tutorial2/Inputs/` before running. The GitHub checkout retains only the path-configuration example in `Inputs/`. Keep `tutorial_utils.py`, which is shared by the notebooks. Analysis outputs remain in `outputs/`, separately for each tutorial. Each notebook includes an Output guide.

## Functions and metrics

`tutorial_utils.py` provides shared input resolution and format checks. It reuses `ScConcept` annotation and coherence functions. The robustness tutorial uses `SingleCellDataset`, as in Tutorial 1. Large sparse matrices are read in the same schema and scored by gene blocks to avoid densifying the entire atlas.

The paper's TC/TD helpers retain the reported background and denominator choices. They are calculated from each current concept set. `ScConcept.evaluation()` uses a different TD definition and caches annotations, so it is not used to evaluate multiple saved runs. Actual NMI and AMI remain separate. Developmental potency retains raw-expression sums and missing-gene masking, following `ScConcept.dannotation()`.

## Analysis-specific dependencies

Use a notebook kernel with the dependencies needed by the enabled sections. These tutorials do not install packages.

| Enabled section | Additional packages |
|---|---|
| Melanoma SHAP/GSEA | `shap`, `gseapy` |
| CellTypist perturbation validation | `celltypist` and its configured model file |
| TCGA survival | `lifelines` |
| Drug ranking from Excel inputs | `openpyxl` |
| Interactive drug–target Sankey | `plotly` |
| Optional Sankey PDF export | `kaleido` with a supported rendering backend |
| Optional new Claude scoring | `anthropic` |

## Data bundle

See [DATA_BUNDLE.md](DATA_BUNDLE.md) for archive selection and extraction instructions. Generated outputs are recreated by running the notebooks and are excluded from the software upload.

### Restore the reference inputs

Download `reference_outputs.zip` from the separately distributed data bundle and extract it into the repository root (the directory containing `scConcept.py`):

```bash
unzip -n /path/to/reference_outputs.zip -d /path/to/scConcept
```

The ZIP contains paths beginning with `Tutorial2/`; do not extract it into `Tutorial2/` itself. It restores the required `Inputs/robustness/`, `Inputs/drug_ranking/`, and `Inputs/transfer/` files. Select the expression-data ZIPs for your analysis as described in `DATA_BUNDLE.md`.

When using the current GitHub checkout, use its notebooks and helper directly. Do not extract the older `tutorial_snapshot.zip`: it retains the historical notebook locations under `Tutorial2/Concept/`. No public data download URL/DOI has been added yet.

## Hardware used for the reported experiments

Most experiments were conducted on a system equipped with an NVIDIA RTX 2080 Ti GPU and 378 GB RAM. For large-scale datasets, including the human lung hierarchical dataset and the non-small cell lung cancer atlas, computations were performed on an NVIDIA A100 GPU (40 GB memory).

These specifications describe the hardware used for the reported experiments, rather than minimum requirements for every tutorial. Software versions for the baseline methods are recorded below.

## Baseline software environments

See [baseline_software_environments.md](baseline_software_environments.md) for the recorded software versions and environment relationships of the ten benchmark baseline methods. This is a software environment record; baseline training pipelines and checkpoints are not included here.
