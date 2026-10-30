# Reference concept files

This directory contains copies of the final GPT-5 concept files selected by the authors for downstream analyses. Source files are preserved. Only filenames were changed; JSON contents, concept order, and gene order are unchanged. Each copied file was checked for byte-for-byte equality with its source. SHA-256 checksums and original paths are recorded in `manifest.csv`.

## Naming

- Transfer: `GPT5_<figure_dataset_name>_concepts_match.json`.
- Benchmark: `GPT5_<figure_dataset_name>_concepts.json`.
- Developmental potency: `GPT5_<manuscript_dataset_name>_concepts.json`.
- Hierarchical analysis: `GPT5_<dataset>_level1_concepts.json` and `GPT5_<dataset>_level2_concepts.json`.
- Melanoma case study: `GPT5_Melanoma_concepts.json` (source suffix `concepts5_nokeep`).
- Large-scale NSCLC atlas: `GPT5_Salcher_concepts.json`.

Benchmark aliases follow the supplied figure: `Qx_Limb_Muscle` → `Schaum_muscle`, `tmuris` → `Schaum_tmuris`, `Vieira_Teichmann_2019` → `Vieira`, and `Reyfman_Misharin_2018` → `Reyfman`. Capitalization of the other benchmark names follows the figure. Melanoma is a separate case study and is not included among the 16 benchmark files.

For developmental potency, Mouse_neurogenesis uses the source suffix `Potency2_nokeep`; the other five datasets use `Potency_nokeep`.

## Inventory

66 JSON files are available: 16 benchmark, 6 developmental-potency, 1 melanoma, 4 hierarchical, 1 large-scale atlas, 6 transfer, 30 robustness-run, and 2 robustness-consensus files.

The author-confirmed pancreas level-2 source is `only_gpt5_Pancreas_10k_concepts7_nokeep_level_pct.json`. It contains 10 parent entries with nested `sub_concepts`; it is a concept-definition file, distinct from the per-cell annotation file `Pancreas_10k_concept7_level.json`. Its nested structure is preserved.

Dataset expression files, raw LLM responses, and interpretability evaluation scores are not included in this directory.

The author-confirmed lung level-2 source is `only_gpt5_Lung_10k_concepts5_nokeep_level_pct.json`, saved as `GPT5_Lung_30k_level2_concepts.json`. This supersedes the previously selected `concepts3_nokeep_level` version. The source JSON structure is preserved.

## Dataset names

Hierarchical lung files use `Lung_30k`; the historical source filenames still contain `Lung_10k`. The large-scale atlas uses `Salcher`, omitting the historical `6k` suffix. These naming changes do not alter the source JSON contents.

Developmental-potency filenames follow the manuscript table, replacing spaces with underscores and omitting parentheses. `dataset_display_name` in the manifest records the manuscript label. Original dataset identifiers and source paths remain available for provenance.

| Manuscript dataset | Concept file |
|---|---|
| Retinal neurons (10x) | `GPT5_Retinal_neurons_10x_concepts.json` |
| Neural crest (Smart-seq2) | `GPT5_Neural_crest_Smart-seq2_concepts.json` |
| Mouse embryo 2 (Smart-seq2) | `GPT5_Mouse_embryo_2_Smart-seq2_concepts.json` |
| Mouse neurogenesis (10x) | `GPT5_Mouse_neurogenesis_10x_concepts.json` |
| Mouse mature neural cell types (10x) | `GPT5_Mouse_mature_neural_cell_types_10x_concepts.json` |
| HSCs and MPPs (inDrop) | `GPT5_HSCs_and_MPPs_inDrop_concepts.json` |

Robustness runs use `GPT5_<dataset>_concepts_<original|conservative|cellidentity>_run<1–5>.json`. Saved consensus files use `GPT5_<dataset>_concepts_consensus.json`. Runs are independent calls with the generation seed fixed at 1.

The six reproduction notebooks are in the parent `Tutorial2/` directory. This directory contains only concept definitions and their documentation/manifest. Required non-concept inputs are in `Tutorial2/Inputs/`.
