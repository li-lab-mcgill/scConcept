# scConcept reference data: manuscript reproduction bundle

This bundle contains the 16 processed benchmark datasets and the inputs for the six completed downstream tutorials. It does not contain the eight-baseline training pipeline or model checkpoints. No input expression matrices were filtered, normalized, truncated, or reformatted during packaging.

## Software and directory layout

Install scConcept from its GitHub repository and use a Python environment with the dependencies required by the enabled analysis sections. The data do not replace the scConcept source code. `tutorial_snapshot.zip` supplies the six completed notebooks, `Tutorial2/tutorial_utils.py`, and their documentation; it does not freeze the unfinished provider-interface code. `CODE_COMPATIBILITY.json` records source checksums at packaging time for provenance.

Extract the dataset, reference-output, shared-resource, and tutorial-snapshot ZIPs into the **root of the scConcept repository**, the directory containing `scConcept.py`. ZIP member paths already begin with `Datasets/` or `Tutorial2/`. Do not extract them into `Tutorial2/Concept`.

Do not copy the original server's `Inputs/data_paths.local.json`. For the published layout, leave this local override absent and do not set `SCCONCEPT_DATA_DIR` or `SCCONCEPT_INPUT_CONFIG` to the original server's locations. Optional custom paths can be configured as described in `Tutorial2/README.md`.

## Which archives do I need?

Every tutorial requires **reference_outputs.zip**, plus the installed scConcept source and the appropriate Python environment. With the current GitHub checkout, use its six notebooks directly in `Tutorial2/` and its `tutorial_utils.py`; do not extract the older `tutorial_snapshot.zip`, whose notebook locations under `Tutorial2/Concept/` are historical. The tutorial snapshot is only a packaging-time reference, not an additional requirement for the current checkout.

| Tutorial | Additional ZIPs |
|---|---|
| Figure 3 melanoma | `data_melanoma.zip`; `optional_models.zip` for enabled CellTypist validation; `data_shared.zip` if TC/TD is enabled |
| Figure 4 hierarchy | `data_hierarchy.zip`, `data_shared.zip` |
| Figure 5 developmental potency | `data_potency.zip` |
| Figure 6 lung cancer | `data_lung.zip`, `data_shared.zip`; drug inputs are already in `reference_outputs.zip` |
| Cross-dataset transfer | `data_transfer.zip` |
| Concept robustness | `data_robustness.zip`, `data_shared.zip` |
| Keep all 16 benchmark datasets | `data_benchmark.zip`; concept definitions are in `reference_outputs.zip` |

`data_benchmark.zip` already contains the Zeisel and Pollen MAT files used in `data_robustness.zip`. They are identical copies provided in a small separate archive for selective download; you do not need both ZIPs for robustness alone.

## CellTypist model: different extraction destination

The completed Figure 3 notebook uses the standard cache path `~/.celltypist/data/models/Immune_All_High.pkl`.

Extract **optional_models.zip into `~/.celltypist/data/`**, rather than the repository root. Its single member is `models/Immune_All_High.pkl`. Alternatively, place the model wherever you prefer and set `CELLTYPIST_MODEL_PATH` in the notebook's configuration cell. The deposited model is for the enabled orthogonal validation section, not for scConcept concept generation.

## Dependencies and reference outputs

- All tutorials require the installed scConcept dependencies. UMAP, Scanpy, and AnnData versions can affect embedding coordinates.
- Figure 3 SHAP/GSEA requires `shap` and `gseapy`; orthogonal validation requires `celltypist`; survival requires `lifelines`.
- Figure 6 drug Excel inputs require `openpyxl`; the Sankey requires `plotly`. Its optional PDF export additionally needs a working `kaleido` renderer.
- Optional new Claude scoring needs `anthropic` and an API key; it is disabled by default. Saved scores suffice for reproducing agreement plots.
- The potency benchmark uses saved CytoTRACE2 predictions. SCISSOR labels and drug-affinity predictions are deposited inputs; the tutorials do not rerun those upstream methods.
- NMI and AMI are stored separately. Robustness plots default to AMI to preserve the historical information-metric values with a corrected label. Both TC backgrounds used by the historical panels are retained explicitly.

The large-atlas and SCISSOR sections also need sufficient RAM and computation time for the large H5AD and primary-tumor subset. Providing the data removes missing-input dependencies; it does not remove those computational requirements.

## Integrity and scope

`MANIFEST.csv` records the archive, extraction destination, relative member path, size, and SHA-256 of every packaged file. `SHA256SUMS.txt` records the SHA-256 of each ZIP. `ARCHIVES.json` reports archive-level sizes and verification results. The packaging process verifies every decompressed member against its source-stream SHA-256 without extracting the large data twice.

The current bundle contains parsed concept sets and coherence ratings; original full API conversations/query logs are not represented as available when they are absent. Original accession, preprocessing, citation, and third-party redistribution metadata remain subject to the authors' final deposit documentation. This is a local preparation bundle, not an already published Zenodo record.
