import os
import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import scanpy as sc
import anndata as ad
import gseapy as gp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from openai import OpenAI

from ECRTM.Runner import Runner
from ECRTM.singlecell_dataset import SingleCellDataset

TOPIC_DISTILLATION_PROMPT_TEMPLATE = """
You are given topics derived from a neural topic model on single-cell RNA-seq.

Each topic includes ONLY:
- A list of the top 100 genes for that topic,
sorted from highest weight to lowest weight in the model.

Here are all topics (JSON list):
{topics_json_str}

TASK:
You must perform *biological topic distillation* by analyzing ONLY the top-100 gene lists.

Specifically:
1. Identify topics whose top genes indicate highly similar biological pathways or cell states.
2. Merge such topics into unified, biologically coherent concepts.
3. Remove topics that are:
- too vague (gene list does not form a coherent module),
- too narrow (dominated by a single outlier gene),
- biologically uninterpretable given the provided gene patterns.

For each FINAL biological concept, output:
- "name": a concise 2–4 word biological label.
* Avoid generic labels like "General Regulation".
* Prefer pathway- or state-specific terms.
- "description": a 10–30 word natural language description
* Summarize the biological meaning of this concept.
* Describe the core pathway, cellular program, or functional state.
* Must be consistent with the concept name and gene list.
* Do NOT mention topic IDs, gene counts, or modeling details.
- "genes": EXACTLY 100 representative genes.
* MUST be chosen from the union of gene lists of the merged source topics.
* MUST reflect a coherent pathway or co-expression program.
* MUST respect the original sorted importance: higher-ranked genes are preferred.
* DO NOT invent gene names.
- "source_topics": list of topic IDs merged into this concept.

OUTPUT FORMAT (STRICT):
Respond ONLY with a valid JSON object in EXACTLY this form:

{{
"concepts": [
{{
"name": "Example Name",
"description": "A concise biological description summarizing the core pathway or cellular program represented by this concept.",
"genes": ["GENE1", "GENE2", "GENE3", ..., "GENE100"],
"source_topics": ["topic_0", "topic_3"]
}}
]
}}

ABSOLUTE RULES:
- DO NOT output anything outside the JSON.
- DO NOT add code fences (such as ```json).
- DO NOT invent genes.
- DO NOT reorder the topics_json_str; use it as provided.
- The final genes must reflect the most informative subset of the top 100 genes per topic.
"""

POTENCY_PROGRAM_PROMPT_TEMPLATE = """
You are given topics derived from a neural topic model on single-cell RNA-seq data.

Each topic includes ONLY:
- A list of the top 100 genes for that topic,
  sorted from highest weight to lowest weight in the model.

Here are all topics (JSON list):
{topics_json_str}

------------------------------------------------------------
TASK: Developmental Potency Program Distillation
------------------------------------------------------------

Your goal is to identify biologically coherent gene programs that reflect
DIFFERENT LEVELS OF CELLULAR DEVELOPMENTAL POTENCY.

You must analyze ONLY the provided top-100 gene lists from the topics.
No external knowledge, marker lists, or invented genes are allowed.

The target developmental potency categories are EXACTLY the following six:

1. Toti.  (totipotent)
2. Pluri. (pluripotent)
3. Multi. (multipotent)
4. Oligo. (oligopotent)
5. Uni.   (unipotent)
6. Diff.  (differentiated)

IMPORTANT:
- These categories define an INTERPRETIVE FRAMEWORK, not guaranteed labels.
- Some categories MAY NOT be detectable in the provided topics.
- If no coherent gene program exists for a category, you MUST output an empty gene list for that category.

------------------------------------------------------------
BIOLOGICAL INTERPRETATION GUIDELINES
------------------------------------------------------------

You should infer developmental potency ONLY from gene patterns such as:
- stemness vs lineage restriction,
- transcriptional plasticity vs functional specialization,
- progenitor-like vs terminal differentiation programs,
- broad developmental regulators vs lineage-specific effector genes.

DO NOT:
- assume cell types,
- assume developmental time points,
- use known CytoTRACE markers explicitly,
- force topics into categories if evidence is weak.

A gene program must show CONSISTENT biological signals across one or more topics
to be assigned to a potency category.

------------------------------------------------------------
PROGRAM CONSTRUCTION RULES
------------------------------------------------------------

For EACH of the six potency categories:

- Identify zero or more topics whose top genes SUPPORT that potency level.
- Extract a representative gene set from those topics.

Gene selection rules:
- Genes MUST come ONLY from the union of genes in the selected source topics.
- Prefer genes that are:
  * highly ranked in their topics,
  * recurrent across topics,
  * biologically consistent with the inferred potency level.
- DO NOT invent gene names.
- Gene lists may have VARIABLE length (including empty lists).

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON)
------------------------------------------------------------

Respond ONLY with a valid JSON object in EXACTLY the following format:

{{
  "concepts": [
    {{
      "name": "Toti.",
      "genes": ["GENE1", "GENE2", "..."],
      "source_topics": ["topic_1", "topic_7"]
    }},
    {{
      "name": "Pluri.",
      "genes": [],
      "source_topics": []
    }},
    {{
      "name": "Multi.",
      "genes": ["GENE_A", "GENE_B"],
      "source_topics": ["topic_3"]
    }},
    {{
      "name": "Oligo.",
      "genes": ["GENE_X", "GENE_Y"],
      "source_topics": ["topic_5", "topic_9"]
    }},
    {{
      "name": "Uni.",
      "genes": ["GENE_M", "GENE_N"],
      "source_topics": ["topic_12"]
    }},
    {{
      "name": "Diff.",
      "genes": ["GENE_D1", "GENE_D2"],
      "source_topics": ["topic_15", "topic_18"]
    }}
  ]
}}

------------------------------------------------------------
ABSOLUTE RULES
------------------------------------------------------------

- DO NOT output anything outside the JSON object.
- DO NOT add code fences (no ```json).
- DO NOT invent genes or modify gene symbols.
- DO NOT force assignments when evidence is weak.
- Empty gene lists are VALID and encouraged when appropriate.
- Interpret potency as a CONTINUUM, not discrete cell types.
"""

HIERARCHICAL_PROMPT_TEMPLATE_FORCE_SPLIT = """
You are given ONE biological concept derived from a neural topic model on single-cell RNA-seq.

IMPORTANT:
This concept has been determined to be HETEROGENEOUS and contains multiple
distinct biological programs. Your task is to split it into meaningful sub-concepts.

The input concept is a JSON object with:
- "name": concept name
- "genes": EXACTLY 100 genes representing this concept (gene program)

Here is the input concept (JSON):
{concept_json}

TASK:
Split this concept into biologically meaningful sub-concepts.

A biological program may also correspond to a distinct cell identity
defined by characteristic combinations of marker genes.
Prefer splits that separate mutually exclusive marker gene groups
indicative of different cell identities.
If mutually exclusive marker gene groups defining distinct cell identities
are present, prioritize identity-based splits over shared functional or
pathway-level programs.

------------------------------------------------------------
SPLIT REQUIREMENTS (MANDATORY)
------------------------------------------------------------

- Produce EXACTLY 2 to 4 sub-concepts.
- EACH sub-concept must represent a SINGLE, coherent biological program.
- Sub-concepts must be generalizable and interpretable.

For EACH sub-concept:
- "name": concise 2–4 word biological label.
- "description": a 10–30 word natural language description
      * Summarize the biological meaning of this sub-concept.
      * Describe the core pathway, cellular program, or functional state.
      * Must be consistent with the sub-concept name and gene list.
      * Do NOT mention modeling details or the parent concept.
- "genes": select between 5 and 20 genes.
- ALL genes MUST be chosen from the parent concept's gene list.
- DO NOT invent genes.
- DO NOT add genes not present in the parent list.

------------------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
------------------------------------------------------------

Respond with EXACTLY one JSON object in the following format:

{{
  "concept_name": "<parent_concept_name>",
  "split": true,
  "sub_concepts": [
    {{
      "name": "<sub_concept_name>",
      "description": "A concise description summarizing the biological program represented by this sub-concept.",
      "genes": ["GENE1", "GENE2", ...]
    }}
  ]
}}

OUTPUT RULES:
- "split" MUST be true.
- "sub_concepts" MUST contain 2-4 items.
- Each "genes" list length MUST be between 5 and 20 (inclusive).
- Every gene MUST appear in the parent concept gene list.
- Respond ONLY with valid JSON.
- Do NOT include explanations, comments, or markdown.
"""


class ScConcept:
    """
    High-level API for the scConcept pipeline.

    Pipeline:
        topic()        -> extract topic genes using ECRTM
        concept()      -> convert topics to concepts using GPT
        annotation()   -> annotate cells using concept gene sets
        evaluation()   -> compute ARI / NMI / TC / TD
        visualization()-> PCA + UMAP visualization
    """

    def __init__(self, data_dir=None, results_dir=None, ecrtm_output_dir=None):
        project_root = Path(__file__).resolve().parent

        self.project_root = project_root
        self.data_dir = Path(data_dir) if data_dir else project_root / "Datasets"
        self.results_dir = Path(results_dir) if results_dir else project_root / "Results"
        self.ecrtm_output_dir = Path(ecrtm_output_dir) if ecrtm_output_dir else project_root / "ECRTM" / "output"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.ecrtm_output_dir.mkdir(parents=True, exist_ok=True)

        # runtime state
        self.dataset_name = None
        self.dataset_handler = None

        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        self.gene_names = None
        self.gene_to_idx = None

        self.topic_file = None
        self.topic_gene_lists = None

        self.concepts = None
        self.concept_file = None

        self.annotation_scores = None
        self.pred_concepts = None

        self.metrics = None
        self.adata_vis = None

        self.dconcepts = None
        self.dconcept_file = None
        self.dconcepts_topk = None
        self.dannotation_scores = None
        self.dannotation_scores_for_pred = None
        self.pred_dconcepts = None

        self.hannotation_df = None
    # =========================================================
    # basic utilities
    # =========================================================
    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def flatten_labels(labels):
        labels = np.array(labels, dtype=object).reshape(-1)
        flat = []
        for x in labels:
            if isinstance(x, np.ndarray):
                if x.size == 1:
                    flat.append(x.item())
                else:
                    flat.append(str(x.tolist()))
            else:
                flat.append(x)
        return np.array(flat)

    def _load_dataset(self, dataset_name, batch_size=512):
        self.dataset_name = dataset_name
        self.dataset_handler = SingleCellDataset(
            dataset_name=dataset_name,
            batch_size=batch_size,
            data_dir=str(self.data_dir),
        )

        self.train_data = self.dataset_handler.train_data
        self.test_data = self.dataset_handler.test_data
        self.train_labels = self.dataset_handler.train_labels
        self.test_labels = self.dataset_handler.test_labels
        self.gene_names = [str(g) for g in self.dataset_handler.gene_names]
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}

    def _dataset_result_dir(self, dataset_name=None):
        dataset_name = dataset_name or self.dataset_name
        out = self.results_dir / dataset_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def load_topic_genes(topic_file):
        topic_gene_lists = []
        with open(topic_file, "r", encoding="utf-8") as f:
            for line in f:
                genes = line.strip().split()
                topic_gene_lists.append(genes)
        return topic_gene_lists

    @staticmethod
    def save_json(obj, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =========================================================
    # topic extraction
    # =========================================================
    def _build_topic_args(
        self,
        dataset="pollen",
        n_topic=50,
        seed=1,
        lr=0.002,
        epochs=500,
        batch_size=512,
        lr_scheduler=True,
        lr_step_size=25,
        sinkhorn_alpha=20,
        OT_max_iter=1000,
        weight_loss_ECR=100,
        eval_step=50,
        dropout=0.0,
        en1_units=200,
        beta_temp=0.2,
        device="cpu",
    ):
        return argparse.Namespace(
            dataset=dataset,
            n_topic=n_topic,
            seed=seed,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_step_size=lr_step_size,
            sinkhorn_alpha=sinkhorn_alpha,
            OT_max_iter=OT_max_iter,
            weight_loss_ECR=weight_loss_ECR,
            eval_step=eval_step,
            dropout=dropout,
            en1_units=en1_units,
            beta_temp=beta_temp,
            data_dir=str(self.data_dir),
            output_dir=str(self.ecrtm_output_dir),
            device=device,
        )

    def topic(
        self,
        dataset,
        n_topic=50,
        seed=1,
        lr=0.002,
        epochs=500,
        batch_size=512,
        lr_scheduler=True,
        lr_step_size=25,
        sinkhorn_alpha=20,
        OT_max_iter=1000,
        weight_loss_ECR=100,
        eval_step=50,
        dropout=0.0,
        en1_units=200,
        beta_temp=0.2,
        topic_epoch=None,
        device=None,
    ):
        """
        Run ECRTM and extract top-gene topics.

        Returns
        -------
        str
            Path to the topic txt file.
        """
        import torch

        self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        args = self._build_topic_args(
            dataset=dataset,
            n_topic=n_topic,
            seed=seed,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            lr_scheduler=lr_scheduler,
            lr_step_size=lr_step_size,
            sinkhorn_alpha=sinkhorn_alpha,
            OT_max_iter=OT_max_iter,
            weight_loss_ECR=weight_loss_ECR,
            eval_step=eval_step,
            dropout=dropout,
            en1_units=en1_units,
            beta_temp=beta_temp,
            device=device,
        )

        self.set_seed(seed)
        args.vocab_size = self.dataset_handler.n_genes

        runner = Runner(args, self.dataset_handler)
        runner.train(self.dataset_handler.train_loader)

        run_name = f"{dataset}_K{n_topic}_seed{seed}"
        topic_dir = self.ecrtm_output_dir / "topics" / run_name

        if topic_epoch is None:
            topic_epoch = epochs

        topic_file = topic_dir / f"epoch{topic_epoch}_top_genes.txt"
        if not topic_file.exists():
            raise FileNotFoundError(f"Topic file not found: {topic_file}")

        self.topic_file = str(topic_file)
        self.topic_gene_lists = self.load_topic_genes(self.topic_file)

        return self.topic_file


    def dataprocess(
        self,
        input_name,
        output_name,
        label_name=None,
        label_key="cell_type",
        target_hvg=10000,
        save_logged_matrix=True,
    ):
        """
        Convert labeled single-cell data into the .mat format required by ECRTM,
        using a preprocessing flow that closely matches the legacy pipeline.

        Supported inputs
        ----------------
        1. h5ad file:
        - input_name = filename of a .h5ad file under self.data_dir
        - labels are read from adata.obs[label_key]

        2. csv + label csv:
        - input_name = count matrix csv filename under self.data_dir
        - label_name = label csv filename under self.data_dir
        - the first column of label csv is used as labels

        Legacy-compatible processing
        ----------------------------
        1. Ensure data is in cells × genes format
        2. Convert gene names to uppercase
        3. Make a copy of the original matrix
        4. Apply log1p on one copy
        5. Select up to target_hvg highly variable genes on the log-transformed copy
        6. Use the same HVG mask to subset the original copy
        7. Save the subsetted matrix to .mat

        Parameters
        ----------
        input_name : str
            Filename of input .h5ad or count .csv under self.data_dir.
        output_name : str
            Name used when saving the output .mat file.
        label_name : str, optional
            Filename of label csv under self.data_dir. Required for csv input.
        label_key : str, default="cell_type"
            Label column in h5ad.obs when input is h5ad.
        target_hvg : int, default=10000
            Number of highly variable genes to keep.
        save_logged_matrix : bool, default=False
            If False, save the original-expression matrix after HVG selection
            (closer to the legacy workflow).
            If True, save the log1p-transformed matrix after HVG selection.

        Returns
        -------
        str
            Path to the saved .mat file.
        """
        import copy
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import anndata as ad
        from scipy import sparse
        from scipy import io as sio

        input_path = self.data_dir / input_name
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # -----------------------------------------------------
        # 1. Load data
        # -----------------------------------------------------
        if input_path.suffix == ".h5ad":
            adata = sc.read_h5ad(input_path)

            if label_key not in adata.obs.columns:
                raise ValueError(
                    f"Label key '{label_key}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

            labels = adata.obs[label_key].astype(str).values

            # make gene names uppercase to match legacy processing
            adata.var_names = pd.Index([str(g).upper() for g in adata.var_names])

            expr = adata.copy()

        elif input_path.suffix == ".csv":
            if label_name is None:
                raise ValueError("label_name must be provided when input_name is a csv file.")

            label_path = self.data_dir / label_name
            if not label_path.exists():
                raise FileNotFoundError(f"Label file not found: {label_path}")

            expr_df = pd.read_csv(input_path, index_col=0)
            label_df = pd.read_csv(label_path, index_col=0)

            if label_df.shape[1] < 1:
                raise ValueError("Label csv must contain at least one column.")

            labels = label_df.iloc[:, 0].astype(str).values

            # legacy code used gene names in uppercase
            expr_df.index = expr_df.index.astype(str).str.upper()
            expr_df.columns = expr_df.columns.astype(str)

            # -------------------------------------------------
            # IMPORTANT:
            # Expect cells × genes.
            # If csv is genes × cells, transpose automatically.
            # -------------------------------------------------
            if expr_df.shape[0] == len(labels):
                # already cells × genes
                pass
            elif expr_df.shape[1] == len(labels):
                # genes × cells -> transpose to cells × genes
                expr_df = expr_df.T
                print("Input count matrix appears to be genes × cells; transposed to cells × genes.")
            else:
                raise ValueError(
                    f"Expression matrix shape {expr_df.shape} does not match label count {len(labels)}. "
                    f"Expected either cells × genes with {len(labels)} rows, "
                    f"or genes × cells with {len(labels)} columns."
                )

            expr = ad.AnnData(expr_df.values)
            expr.var_names = pd.Index(expr_df.columns.astype(str))
            expr.obs_names = pd.Index(expr_df.index.astype(str))

        else:
            raise ValueError("Unsupported input format. Please provide a .h5ad or .csv file.")

        if expr.shape[0] != len(labels):
            raise ValueError(
                f"Number of cells in expression matrix ({expr.shape[0]}) "
                f"does not match number of labels ({len(labels)})."
            )

        print(f"Raw data shape: {expr.shape}")

        # -----------------------------------------------------
        # Create two copies
        #    - expr_hvg: for log1p + HVG selection
        #    - expr_raw: original-expression copy for final saving
        # -----------------------------------------------------
        expr_hvg = expr.copy()
        expr_raw = expr.copy()

        # -----------------------------------------------------
        # log1p on the HVG-selection copy
        #    This matches your old order: log1p -> HVG
        # -----------------------------------------------------
        sc.pp.log1p(expr_hvg)

        # -----------------------------------------------------
        # HVG selection
        # -----------------------------------------------------
        n_genes = expr_hvg.shape[1]

        if n_genes > target_hvg:
            sc.pp.highly_variable_genes(
                expr_hvg,
                n_top_genes=target_hvg
            )
            hv_genes_mask = expr_hvg.var["highly_variable"].copy()

            expr_hvg = expr_hvg[:, hv_genes_mask].copy()
            expr_raw = expr_raw[:, hv_genes_mask].copy()

            print(f"Selected top {expr_hvg.shape[1]} highly variable genes.")
        else:
            print(f"Gene number ({n_genes}) <= {target_hvg}, keeping all genes.")

        # -----------------------------------------------------
        # Choose which matrix to save
        #    legacy-like default: save original-expression matrix
        # -----------------------------------------------------
        expr_to_save = expr_hvg if save_logged_matrix else expr_raw

        print(f"Processed data shape: {expr_to_save.shape}")
        print("Using full dataset for both training and evaluation (no train/test split).")

        # -----------------------------------------------------
        # Convert to sparse CSR matrix
        # -----------------------------------------------------
        X = expr_to_save.X
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = sparse.csr_matrix(np.asarray(X, dtype=np.float32))

        gene_names = np.array(expr_to_save.var_names.tolist(), dtype=object).reshape(1, -1)

        # -----------------------------------------------------
        # No train/test split
        # -----------------------------------------------------
        bow_train = X
        bow_test = X

        label_train = np.asarray(labels)
        label_test = np.asarray(labels)

        # Dummy docs for compatibility
        doc_train = np.array([" "] * bow_train.shape[0], dtype=object).reshape(-1, 1)
        doc_test = np.array([" "] * bow_test.shape[0], dtype=object).reshape(-1, 1)

        # -----------------------------------------------------
        # Save .mat
        # -----------------------------------------------------
        output_mat_path = self.data_dir / f"{output_name}.mat"
        output_mat_path.parent.mkdir(parents=True, exist_ok=True)

        sio.savemat(output_mat_path, {
            "bow_train": bow_train,
            "bow_test": bow_test,
            "voc": gene_names,
            "label_train": label_train.reshape(-1),
            "label_test": label_test.reshape(-1),
            "doc_train": doc_train,
            "doc_test": doc_test,
        })

        print(f"Saved processed dataset to: {output_mat_path}")

        return str(output_mat_path)
    # =========================================================
    # concept generation
    # =========================================================
    def _build_topic_distillation_prompt(self, topic_gene_lists):
        import json

        topics_json = []
        for i, genes in enumerate(topic_gene_lists):
            topics_json.append({
                "topic_id": f"topic_{i}",
                "genes": genes
            })

        topics_json_str = json.dumps(topics_json, indent=2)

        # 🔥 关键改动：用 .format() 填充
        prompt = TOPIC_DISTILLATION_PROMPT_TEMPLATE.format(
            topics_json_str=topics_json_str
        )

        return prompt.strip()

    def concept(
        self,
        topic_file,
        api_key,
        model="gpt-5",
        seed=1,
        temperature=0.2,
        save_name=None,
        system_prompt="You are an expert in single-cell biology."
    ):
        """
        Generate concepts from topic top-gene file using GPT.

        Returns
        -------
        list[dict]
            Concept list.
        """
        self.topic_file = str(topic_file)
        self.topic_gene_lists = self.load_topic_genes(self.topic_file)

        topic_distillation_prompt = self._build_topic_distillation_prompt(self.topic_gene_lists)

        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": topic_distillation_prompt}
            ]
        )

        response_text = completion.choices[0].message.content
        parsed = json.loads(response_text)
        concepts = parsed["concepts"]

        self.concepts = concepts

        dataset_dir = self._dataset_result_dir(self.dataset_name or "unknown_dataset")
        save_name = save_name or f"{model}_{self.dataset_name}_concepts.json"
        concept_file = dataset_dir / save_name
        self.save_json(concepts, concept_file)

        self.concept_file = str(concept_file)
        return concepts

    # =========================================================
    # annotation
    # =========================================================
    @staticmethod
    def generate_topk_concepts(concepts, k):
        new_concepts = []
        for concept in concepts:
            new_concepts.append({
                "name": concept["name"],
                "genes": concept["genes"][:k],
                "source_topics": concept.get("source_topics", [])
            })
        return new_concepts

    @staticmethod
    def assign_cells_by_concepts_zscore(concepts, test_data, gene_to_idx):
        expr = test_data.detach().cpu().numpy().astype(float)
        num_cells, num_genes = expr.shape

        gene_mean = expr.mean(axis=0, keepdims=True)
        gene_std = expr.std(axis=0, keepdims=True) + 1e-6
        expr_z = (expr - gene_mean) / gene_std

        concept_names = [c["name"] for c in concepts]
        scores = np.zeros((num_cells, len(concept_names)), dtype=float)

        for concept_idx, concept in enumerate(concepts):
            marker_genes = concept["genes"]
            marker_indices = [gene_to_idx[g] for g in marker_genes if g in gene_to_idx]
            if len(marker_indices) == 0:
                continue
            scores[:, concept_idx] = expr_z[:, marker_indices].mean(axis=1)

        max_indices = np.argmax(scores, axis=1)
        pred_labels = [concept_names[i] for i in max_indices]
        return scores, concept_names, pred_labels

    def annotation(self, concepts=None, concept_file=None, dataset=None, batch_size=500, topk=100):
        """
        Annotate cells using concept gene sets.

        Returns
        -------
        tuple
            scores, concept_names, pred_labels
        """
        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if concepts is None:
            if concept_file is not None:
                concepts = self.load_json(concept_file)
            elif self.concepts is not None:
                concepts = self.concepts
            else:
                raise ValueError("No concepts provided. Please pass concepts or concept_file.")

        new_concepts = self.generate_topk_concepts(concepts, k=topk)

        scores, concept_names, pred_labels = self.assign_cells_by_concepts_zscore(
            new_concepts,
            self.test_data,
            self.gene_to_idx
        )

        self.concepts = new_concepts
        self.annotation_scores = scores
        self.pred_concepts = pred_labels

        dataset_dir = self._dataset_result_dir(self.dataset_name)
        self.save_json(pred_labels, dataset_dir / f"{self.dataset_name}_pred_concepts.json")
        np.save(dataset_dir / f"{self.dataset_name}_concept_scores.npy", scores)

        return scores, concept_names, pred_labels

    # =========================================================
    # evaluation
    # =========================================================
    @staticmethod
    def purity_score(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        contingency = pd.crosstab(pd.Series(y_true, name="true"),
                                  pd.Series(y_pred, name="pred"))
        return contingency.max(axis=0).sum() / contingency.values.sum()

    @staticmethod
    def topk_genes_nested(concepts, k=10):
        nested = []
        for concept in concepts:
            genes = concept.get("genes", [])
            nested.append([str(g).upper() for g in genes[:k]])
        return nested

    @staticmethod
    def compute_coherence_from_gene_lists(doc_word, vocab, concept_gene_lists, dicts_gene_tran):
        N = 10
        topic_size = len(concept_gene_lists)
        doc_size = doc_word.shape[0]

        gene2idx = {g: i for i, g in enumerate(vocab)}

        topic_list = []
        for genes in concept_gene_lists:
            arr_idx = []
            for g in genes:
                idx = gene2idx.get(g, -1)
                arr_idx.append(idx)

            if len(arr_idx) > N:
                arr_idx = arr_idx[:N]
            elif len(arr_idx) < N:
                arr_idx = arr_idx + [-1] * (N - len(arr_idx))

            topic_list.append(np.array(arr_idx, dtype=int))

        sum_coherence_score = 0.0

        for i in range(topic_size):
            word_array = topic_list[i]
            sum_score = 0.0

            for n in range(N):
                wn = word_array[n]
                if wn in dicts_gene_tran:
                    flag_n = doc_word[:, dicts_gene_tran[wn]] > 0
                    p_n = np.sum(flag_n) / doc_size

                    for l in range(n + 1, N):
                        wl = word_array[l]
                        if wl in dicts_gene_tran:
                            flag_l = doc_word[:, dicts_gene_tran[wl]] > 0
                            p_l = np.sum(flag_l)
                            p_nl = np.sum(flag_n * flag_l)

                            if p_n * p_l * p_nl > 0:
                                p_l = p_l / doc_size
                                p_nl = p_nl / doc_size

                                if (p_l * p_n) == 1 and p_nl == 1:
                                    continue

                                sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)

            sum_coherence_score += sum_score * (2 / (N * N - N))

        return sum_coherence_score / topic_size

    @staticmethod
    def topic_diversity(concept_gene_lists, topk=10):
        all_genes = []
        for genes in concept_gene_lists:
            all_genes.extend(genes[:topk])
        unique_genes = len(set(all_genes))
        total_genes = len(concept_gene_lists) * topk
        return unique_genes / total_genes if total_genes > 0 else 0.0

    def evaluation(
        self,
        concepts=None,
        concept_file=None,
        dataset=None,
        batch_size=500,
        topk_annotation=100,
        species="human",
        gmt_path=None
    ):
        """
        Compute ARI / NMI / TC / TD.

        Parameters
        ----------
        concepts : list[dict], optional
            Concept list. If not provided, concepts will be loaded from
            `concept_file` or from `self.concepts`.
        concept_file : str, optional
            Path to concept JSON file.
        dataset : str, optional
            Dataset name. If provided and different from the currently loaded
            dataset, the dataset will be reloaded.
        batch_size : int, default=500
            Batch size used when loading the dataset.
        topk_annotation : int, default=100
            Number of top genes per concept used for cell annotation.
        species : {"human", "mouse"}, default="human"
            Species used to select the default MSigDB GMT file when `gmt_path`
            is not provided.
            - "human" -> msigdb.v2024.1.Hs.symbols.gmt
            - "mouse" -> msigdb.v2024.1.Mm.symbols.gmt
        gmt_path : str, optional
            Path to a custom GMT file, or a GMT filename located under `self.data_dir`.
            If provided, it overrides the `species` argument.

        Returns
        -------
        dict
            Dictionary containing:
            - Purity
            - ARI
            - NMI
            - TC
            - TD
        """
        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if self.pred_concepts is None or self.annotation_scores is None:
            self.annotation(
                concepts=concepts,
                concept_file=concept_file,
                dataset=dataset,
                batch_size=batch_size,
                topk=topk_annotation
            )

        true_labels = self.flatten_labels(self.test_labels)
        pred_labels = np.array(self.pred_concepts)

        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        purity = self.purity_score(true_labels, pred_labels)

        # -----------------------------------------------------
        # Load concepts for TC / TD
        # -----------------------------------------------------
        if concepts is None:
            if concept_file is not None:
                concepts = self.load_json(concept_file)
            elif self.concepts is not None:
                concepts = self.concepts
            else:
                raise ValueError("No concepts available for evaluation.")

        nested_gene_lists = self.topk_genes_nested(concepts, k=10)

        # -----------------------------------------------------
        # Resolve GMT path
        # Priority:
        #   1) user-provided gmt_path
        #   2) species-based default
        # -----------------------------------------------------
        if gmt_path is not None:
            gmt_path = Path(gmt_path)
            if not gmt_path.is_absolute():
                gmt_path = self.data_dir / gmt_path
        else:
            species = species.lower()
            if species in ["human", "hs", "homo_sapiens", "homo sapiens"]:
                gmt_filename = "msigdb.v2024.1.Hs.symbols.gmt"
            elif species in ["mouse", "mm", "mus_musculus", "mus musculus"]:
                gmt_filename = "msigdb.v2024.1.Mm.symbols.gmt"
            else:
                raise ValueError(
                    f"Unsupported species: {species}. "
                    f"Please use 'human', 'mouse', or provide gmt_path explicitly."
                )
            gmt_path = self.data_dir / gmt_filename

        if not gmt_path.exists():
            raise FileNotFoundError(f"GMT file not found: {gmt_path}")

        # -----------------------------------------------------
        # Build background matrix from GMT
        # -----------------------------------------------------
        kegg = gp.read_gmt(path=str(gmt_path))
        bg_vocab = sorted(list(set(g.upper() for genes in kegg.values() for g in genes)))
        bg_gene_to_idx = {g: i for i, g in enumerate(bg_vocab)}

        bg_data = np.zeros((len(kegg), len(bg_vocab)), dtype=float)
        for pathway_idx, genes in enumerate(kegg.values()):
            for g in genes:
                g = g.upper()
                bg_data[pathway_idx, bg_gene_to_idx[g]] = 1.0

        # IMPORTANT:
        # compute_coherence_from_gene_lists expects:
        #   vocab index -> doc_word column index
        dicts_gene_tran = {i: i for i in range(len(bg_vocab))}

        tc = self.compute_coherence_from_gene_lists(
            doc_word=bg_data,
            vocab=bg_vocab,
            concept_gene_lists=nested_gene_lists,
            dicts_gene_tran=dicts_gene_tran
        )

        td = self.topic_diversity(nested_gene_lists, topk=10)

        metrics = {
            "Purity": float(purity),
            "ARI": float(ari),
            "NMI": float(nmi),
            "TC": float(tc),
            "TD": float(td),
            "GMT": str(gmt_path),
        }

        self.metrics = metrics

        dataset_dir = self._dataset_result_dir(self.dataset_name)
        self.save_json(metrics, dataset_dir / f"{self.dataset_name}_evaluation.json")

        return metrics

    # =========================================================
    # visualization
    # =========================================================
    def visualization(
        self,
        dataset=None,
        batch_size=500,
        pred_concepts=None,
        save_name=None,
        n_pcs=50,
        n_neighbors=15,
        point_size=None,
    ):
        """
        Visualize cell types and predicted concepts with PCA + UMAP.

        Returns
        -------
        AnnData
        """
        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if pred_concepts is None:
            if self.pred_concepts is None:
                raise ValueError("No predicted concepts available. Run annotation() first.")
            pred_concepts = self.pred_concepts

        X = self.test_data.detach().cpu().numpy()
        cell_types = self.flatten_labels(self.test_labels)
        concept_labels = np.array(pred_concepts)

        adata_vis = ad.AnnData(X)
        adata_vis.obs["cell_type"] = pd.Categorical(cell_types)
        adata_vis.obs["concept"] = pd.Categorical(concept_labels)
        adata_vis.var_names = [str(g) for g in self.gene_names]

        sc.pp.pca(adata_vis, n_comps=n_pcs)
        sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(adata_vis)

        fig = sc.pl.umap(
            adata_vis,
            color=["cell_type", "concept"],
            ncols=2,
            frameon=False,
            wspace=0.35,
            legend_loc="right margin",
            size=point_size,
            return_fig=True
        )

        for ax in fig.axes[:2]:
            if ax.legend_ is not None:
                for text in ax.legend_.get_texts():
                    text.set_fontsize(12)

        dataset_dir = self._dataset_result_dir(self.dataset_name)
        save_name = save_name or f"{self.dataset_name}_umap_celltype_concept.pdf"
        save_path = dataset_dir / save_name
        fig.savefig(save_path, bbox_inches="tight")

        self.adata_vis = adata_vis
        return adata_vis

    def _build_potency_prompt(self, topic_gene_lists):
        import json

        topics_json = []
        for i, genes in enumerate(topic_gene_lists):
            topics_json.append({
                "topic_id": f"topic_{i}",
                "genes": genes
            })

        topics_json_str = json.dumps(topics_json, indent=2)

        prompt = POTENCY_PROGRAM_PROMPT_TEMPLATE.format(
            topics_json_str=topics_json_str
        )

        return prompt.strip()

    def dconcept(
        self,
        topic_file,
        api_key,
        model="gpt-5",
        seed=1,
        save_name=None,
        system_prompt="You are an expert in single-cell biology."
    ):
        """
        Extract developmental potency concepts from topic gene lists.

        Parameters
        ----------
        topic_file : str
            Path to the topic top-gene txt file.
        api_key : str
            OpenAI API key.
        model : str, default="gpt-5"
            GPT model name.
        seed : int, default=1
            Random seed for GPT call.
        save_name : str, optional
            Output JSON filename. If None, a default name will be used.
        system_prompt : str, default="You are an expert in single-cell biology."
            System prompt for the GPT call.

        Returns
        -------
        list[dict]
            Developmental potency concept list.
        """
        import json
        from openai import OpenAI

        # -----------------------------------------------------
        # load topics
        # -----------------------------------------------------
        self.topic_file = str(topic_file)
        self.topic_gene_lists = self.load_topic_genes(self.topic_file)

        # -----------------------------------------------------
        # build potency prompt
        # -----------------------------------------------------
        potency_prompt = self._build_potency_prompt(self.topic_gene_lists)

        # -----------------------------------------------------
        # GPT call
        # -----------------------------------------------------
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": potency_prompt}
            ]
        )

        # -----------------------------------------------------
        # parse output
        # -----------------------------------------------------
        response_text = completion.choices[0].message.content.strip()

        # tolerate ```json ... ```
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            response_text = response_text.replace("json", "").strip()

        parsed = json.loads(response_text)
        concepts = parsed["concepts"]

        # -----------------------------------------------------
        # save
        # -----------------------------------------------------
        dataset_dir = self._dataset_result_dir(self.dataset_name or "unknown_dataset")

        save_name = save_name or f"{self.dataset_name}_potency_concepts.json"
        concept_file = dataset_dir / save_name

        self.save_json(concepts, concept_file)

        # -----------------------------------------------------
        # cache
        # -----------------------------------------------------
        self.dconcepts = concepts
        self.dconcept_file = str(concept_file)

        return concepts

    def dannotation(
        self,
        concepts=None,
        dataset=None,
        batch_size=500,
        topk=100,
    ):
        """
        Annotate cells using developmental potency concepts.

        This method uses the expression matrix directly:
        - no z-score normalization
        - score for each concept = sum of expression of concept genes

        Concepts with no matched genes in the dataset are masked so that they
        will not be selected during prediction.

        Parameters
        ----------
        concepts : list[dict], optional
            Developmental concepts returned by `dconcept()`.
            If None, `self.dconcepts` will be used.
        dataset : str, optional
            Dataset name. If provided and different from the currently loaded
            dataset, the dataset will be reloaded.
        batch_size : int, default=500
            Batch size used when loading dataset if needed.
        topk : int, default=100
            Keep top-k genes for each concept before annotation.

        Returns
        -------
        scores : np.ndarray
            Cell × concept score matrix.
        concept_names : list[str]
            Developmental concept names.
        pred_labels : list[str]
            Predicted developmental concept label for each cell.
        """
        import numpy as np

        # -----------------------------------------------------
        # load dataset if needed
        # -----------------------------------------------------
        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if self.test_data is None or self.gene_names is None:
            raise ValueError("Dataset not loaded. Please provide dataset=... or load data first.")

        # -----------------------------------------------------
        # load concepts if not explicitly provided
        # -----------------------------------------------------
        if concepts is None:
            if self.dconcepts is not None:
                concepts = self.dconcepts
            else:
                raise ValueError(
                    "No developmental concepts available. "
                    "Run dconcept() first or pass concepts explicitly."
                )

        # -----------------------------------------------------
        # gene -> index
        # -----------------------------------------------------
        gene_to_idx = {str(g): i for i, g in enumerate(self.gene_names)}

        # -----------------------------------------------------
        # keep top-k genes for each concept
        # -----------------------------------------------------
        new_concepts = self.generate_topk_concepts(concepts, k=topk)

        # -----------------------------------------------------
        # expression matrix (no z-score)
        # -----------------------------------------------------
        expr = self.test_data.detach().cpu().numpy().astype(float)
        num_cells, num_genes = expr.shape

        concept_names = [c["name"] for c in new_concepts]
        num_concepts = len(new_concepts)

        scores = np.zeros((num_cells, num_concepts), dtype=float)

        # -----------------------------------------------------
        # score = sum of expression of concept genes
        # -----------------------------------------------------
        n_found = []

        for ci, concept in enumerate(new_concepts):
            markers = concept["genes"]
            marker_idx = [gene_to_idx[g] for g in markers if g in gene_to_idx]
            n_found.append(len(marker_idx))

            if len(marker_idx) == 0:
                continue

            scores[:, ci] = expr[:, marker_idx].sum(axis=1)

        n_found = np.array(n_found)

        # -----------------------------------------------------
        # mask unsupported concepts
        # -----------------------------------------------------
        valid_mask = n_found > 0

        scores_for_pred = scores.copy()
        scores_for_pred[:, ~valid_mask] = -np.inf

        if np.all(~valid_mask):
            raise ValueError(
                "None of the developmental concepts have genes present in the dataset."
            )

        # -----------------------------------------------------
        # predict by argmax
        # -----------------------------------------------------
        max_idx = np.argmax(scores_for_pred, axis=1)
        pred_labels = [concept_names[i] for i in max_idx]

        # -----------------------------------------------------
        # cache
        # -----------------------------------------------------
        self.dconcepts_topk = new_concepts
        self.dannotation_scores = scores
        self.dannotation_scores_for_pred = scores_for_pred
        self.pred_dconcepts = pred_labels

        # -----------------------------------------------------
        # save
        # -----------------------------------------------------
        dataset_dir = self._dataset_result_dir(self.dataset_name)

        self.save_json(
            pred_labels,
            dataset_dir / f"{self.dataset_name}_pred_dconcepts.json"
        )
        np.save(
            dataset_dir / f"{self.dataset_name}_dconcept_scores.npy",
            scores
        )
        np.save(
            dataset_dir / f"{self.dataset_name}_dconcept_scores_for_pred.npy",
            scores_for_pred
        )

        return scores, concept_names, pred_labels

    def pct_should_split_concept_expr_matrix(
        self,
        X,
        vocab,
        concepts,
        score_old,
        concept_idx,
        tau_frac=0.5,
        min_cells=30,
        min_leaf=20,
        min_impurity_reduction=0.1,
        random_state=0,
    ):
        """
        Expression-based PCT-style split test using expression matrix only.
        """
        from sklearn.cluster import KMeans
        import numpy as np

        info = {"concept_idx": int(concept_idx)}
        gene_to_idx = {g: i for i, g in enumerate(vocab)}

        s_all = score_old[:, concept_idx]
        tau = tau_frac * float(np.max(s_all))
        cell_idx = np.where(s_all > tau)[0]

        n_cells = len(cell_idx)
        info.update({"tau": tau, "n_cells": n_cells})

        if n_cells < min_cells:
            info["reason"] = "too_few_cells"
            return False, info

        concept_genes = concepts[concept_idx]["genes"]
        gene_indices = [gene_to_idx[g] for g in concept_genes if g in gene_to_idx]
        info["n_genes_in_data"] = len(gene_indices)

        if len(gene_indices) < 10:
            info["reason"] = "too_few_genes"
            return False, info

        if hasattr(X, "detach"):
            X_sub = X[cell_idx][:, gene_indices].detach().cpu().numpy()
        else:
            X_sub = X[np.ix_(cell_idx, gene_indices)]

        X_sub = X_sub.astype(float)

        mu = X_sub.mean(axis=0, keepdims=True)
        parent_sse = float(np.sum((X_sub - mu) ** 2))
        info["parent_sse"] = parent_sse

        if parent_sse < 1e-8:
            info["reason"] = "near_zero_variance"
            return False, info

        km = KMeans(n_clusters=2, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_sub)

        idx0 = labels == 0
        idx1 = labels == 1
        n0, n1 = idx0.sum(), idx1.sum()
        info["cluster_sizes"] = (int(n0), int(n1))

        if min(n0, n1) < min_leaf:
            info["reason"] = "small_child_cluster"
            return False, info

        mu0 = X_sub[idx0].mean(axis=0, keepdims=True)
        mu1 = X_sub[idx1].mean(axis=0, keepdims=True)

        sse0 = float(np.sum((X_sub[idx0] - mu0) ** 2))
        sse1 = float(np.sum((X_sub[idx1] - mu1) ** 2))
        child_sse = sse0 + sse1

        impurity_reduction = (parent_sse - child_sse) / parent_sse
        info["impurity_reduction"] = float(impurity_reduction)

        if impurity_reduction < min_impurity_reduction:
            info["reason"] = "low_impurity_reduction"
            return False, info

        info["reason"] = "PASS"
        return True, info


    @staticmethod
    def keep_concept_as_is(concept):
        """
        Wrap an unsplit concept into the same output format as GPT split results.
        """
        return {
            "concept_name": concept.get("name", None),
            "split": False,
            "sub_concepts": [
                {
                    "name": concept.get("name", None),
                    "description": concept.get("description", ""),
                    "genes": concept["genes"],
                }
            ]
        }


    def refine_concept_with_gpt(self, concept, api_key, model="gpt-5", seed=1):
        """
        Split one concept into sub-concepts using the fixed hierarchical prompt.
        """
        import json
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        concept_json_str = json.dumps(
            concept,
            ensure_ascii=False,
            indent=None
        )

        prompt = HIERARCHICAL_PROMPT_TEMPLATE_FORCE_SPLIT.format(
            concept_json=concept_json_str
        )

        resp = client.chat.completions.create(
            model=model,
            seed=seed,
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON. No explanations. No code fences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        output_text = resp.choices[0].message.content.strip()

        if output_text.startswith("```"):
            output_text = output_text.strip("`")
            output_text = output_text.replace("json", "").strip()

        output_json = json.loads(output_text)
        return output_json


    @staticmethod
    def flatten_subconcepts_minimal(refined_results):
        """
        Flatten hierarchical GPT results into one flat concept list.

        Returns
        -------
        flat_concepts : list[dict]
        parent_ids_for_new : list[int]
            parent concept index for each flattened sub-concept
        """
        flat_concepts = []
        parent_ids_for_new = []

        for parent_idx, item in enumerate(refined_results):
            sub_concepts = item.get("sub_concepts", [])
            for sub in sub_concepts:
                flat_concepts.append({
                    "name": sub["name"],
                    "description": sub.get("description", ""),
                    "genes": sub["genes"],
                    "source_topics": []
                })
                parent_ids_for_new.append(parent_idx)

        return flat_concepts, parent_ids_for_new

    def hconcept(
        self,
        concepts,
        scores,
        dataset=None,
        batch_size=500,
        api_key=None,
        model="gpt-5",
        seed=1,
        tau_frac=0.5,
        min_cells=30,
        min_leaf=20,
        min_impurity_reduction=0.1,
        random_state=1,
        save_name=None,
    ):
        """
        Build second-layer hierarchical concepts from first-layer concepts.

        Workflow
        --------
        1. For each first-layer concept, use the PCT-style expression split test
        to decide whether it should be subdivided.
        2. If split is needed, call GPT with the fixed hierarchical prompt.
        3. Otherwise, keep the concept as is.
        4. Flatten all sub-concepts into a second-layer concept list.

        Parameters
        ----------
        concepts : list[dict]
            First-layer concepts.
        scores : np.ndarray
            Cell × first-layer concept score matrix.
        dataset : str, optional
            Dataset name. Used to load test_data and gene_names if needed.
        batch_size : int, default=500
            Batch size for dataset loading.
        api_key : str, optional
            OpenAI API key. Required if any concept needs GPT-based split.
        model : str, default="gpt-5"
            GPT model for hierarchical split.
        seed : int, default=1
            Random seed for GPT calls.
        tau_frac, min_cells, min_leaf, min_impurity_reduction, random_state
            Hyperparameters for PCT split decision.
        save_name : str, optional
            Output JSON filename for hierarchical concepts.

        Returns
        -------
        refined_results : list[dict]
            Per-parent split results.
        flat_concepts : list[dict]
            Flattened second-layer sub-concepts.
        """
        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if self.test_data is None or self.gene_names is None:
            raise ValueError("Dataset not loaded. Please provide dataset=... or load data first.")

        X = self.test_data
        vocab = self.gene_names
        score_old = scores

        if score_old.shape[0] != X.shape[0]:
            raise ValueError("Cell number mismatch between scores and expression matrix.")
        if score_old.shape[1] != len(concepts):
            raise ValueError("Concept number mismatch between scores and concept list.")

        refined_results = []
        pct_logs = []

        for idx, concept in enumerate(concepts):
            should_split, info = self.pct_should_split_concept_expr_matrix(
                X=X,
                vocab=vocab,
                concepts=concepts,
                score_old=score_old,
                concept_idx=idx,
                tau_frac=tau_frac,
                min_cells=min_cells,
                min_leaf=min_leaf,
                min_impurity_reduction=min_impurity_reduction,
                random_state=random_state,
            )

            pct_logs.append({
                "concept_idx": idx,
                "should_split": should_split,
                "info": info,
            })

            if should_split:
                if api_key is None:
                    raise ValueError(
                        "api_key is required when a concept needs GPT-based hierarchical split."
                    )
                refined = self.refine_concept_with_gpt(
                    concept=concept,
                    api_key=api_key,
                    model=model,
                    seed=seed,
                )
            else:
                refined = self.keep_concept_as_is(concept)

            refined_results.append(refined)

        flat_concepts, parent_ids_for_new = self.flatten_subconcepts_minimal(refined_results)

        # cache
        self.hconcepts = flat_concepts
        self.hconcept_refined_results = refined_results
        self.hconcept_parent_ids = parent_ids_for_new
        self.hconcept_pct_logs = pct_logs

        # save
        dataset_dir = self._dataset_result_dir(self.dataset_name)

        hierarchical_json = {
            "refined_results": refined_results,
            "flat_concepts": flat_concepts,
            "parent_ids_for_new": parent_ids_for_new,
            "pct_logs": pct_logs,
        }

        save_name = save_name or f"{self.dataset_name}_hierarchical_concepts.json"

        self.save_json(
            hierarchical_json,
            dataset_dir / save_name
        )

        self.hconcept_file = str(dataset_dir / save_name)

        return refined_results, flat_concepts

    def hannotation(
        self,
        concepts=None,
        dataset=None,
        batch_size=500,
        topk=100,
    ):
        """
        Annotate cells using second-layer hierarchical concepts.

        This method uses the standard concept annotation logic:
        - gene-wise z-score normalization
        - score for each concept = mean z-score of concept genes

        Parameters
        ----------
        concepts : list[dict], optional
            Flattened second-layer concepts.
            If None, `self.hconcepts` will be used.
        dataset : str, optional
            Dataset name. If provided and different from the currently loaded
            dataset, the dataset will be reloaded.
        batch_size : int, default=500
            Batch size used when loading dataset if needed.
        topk : int, default=100
            Keep top-k genes for each concept before annotation.

        Returns
        -------
        scores : np.ndarray
            Cell × hierarchical concept score matrix.
        concept_names : list[str]
            Hierarchical concept names.
        pred_labels : list[str]
            Predicted hierarchical concept label for each cell.
        """
        import numpy as np

        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if self.test_data is None or self.gene_names is None:
            raise ValueError("Dataset not loaded. Please provide dataset=... or load data first.")

        if concepts is None:
            if getattr(self, "hconcepts", None) is not None:
                concepts = self.hconcepts
            else:
                raise ValueError(
                    "No hierarchical concepts available. Run hconcept() first or pass concepts explicitly."
                )

        gene_to_idx = {str(g): i for i, g in enumerate(self.gene_names)}

        new_concepts = self.generate_topk_concepts(concepts, k=topk)

        # same logic as standard annotation
        scores, concept_names, pred_labels = self.assign_cells_by_concepts_zscore(
            new_concepts,
            self.test_data,
            gene_to_idx
        )

        # cache
        self.hconcepts_topk = new_concepts
        self.hannotation_scores = scores
        self.pred_hconcepts = pred_labels

        # save
        dataset_dir = self._dataset_result_dir(self.dataset_name)

        self.save_json(
            pred_labels,
            dataset_dir / f"{self.dataset_name}_hierarchical_pred_labels.json"
        )
        np.save(
            dataset_dir / f"{self.dataset_name}_hierarchical_scores.npy",
            scores
        )

        return scores, concept_names, pred_labels

    def hierarchical_assign(
        self,
        score_old,
        score_new,
        parent_ids_for_new,
        old_concept_names=None,
        new_concept_names=None,
        old_threshold=None,
        new_threshold=None,
    ):
        """
        Hierarchical assignment using:
        - first-layer concept scores
        - second-layer concept scores
        - parent-child mapping
        """
        import numpy as np
        import pandas as pd

        n_cells, n_old = score_old.shape
        _, n_new = score_new.shape

        parent_ids_for_new = np.asarray(parent_ids_for_new)
        assert parent_ids_for_new.shape[0] == n_new, "parent_ids_for_new length must equal n_new"

        # first choose the best parent concept
        old_best_idx = np.argmax(score_old, axis=1)
        old_best_score = score_old[np.arange(n_cells), old_best_idx]

        new_best_idx = np.full(n_cells, -1, dtype=int)
        new_best_score = np.full(n_cells, np.nan)

        for i in range(n_cells):
            parent = old_best_idx[i]

            if (old_threshold is not None) and (old_best_score[i] < old_threshold):
                continue

            child_mask = (parent_ids_for_new == parent)
            child_indices = np.where(child_mask)[0]

            if child_indices.size == 0:
                continue

            scores_for_children = score_new[i, child_indices]
            local_best_pos = np.argmax(scores_for_children)
            child_global_idx = child_indices[local_best_pos]
            child_score = scores_for_children[local_best_pos]

            if (new_threshold is not None) and (child_score < new_threshold):
                continue

            new_best_idx[i] = child_global_idx
            new_best_score[i] = child_score

        df = pd.DataFrame({
            "old_concept_idx": old_best_idx,
            "score_old_max": old_best_score,
            "new_concept_idx": new_best_idx,
            "score_new_max_in_parent": new_best_score,
        })

        if old_concept_names is not None:
            df["old_concept_name"] = [old_concept_names[idx] for idx in old_best_idx]

        if new_concept_names is not None:
            name_col = []
            for idx in new_best_idx:
                if idx == -1:
                    name_col.append(None)
                else:
                    name_col.append(new_concept_names[idx])
            df["new_concept_name"] = name_col

        return df

    def hannotation(
        self,
        concepts=None,
        scores_old=None,
        dataset=None,
        batch_size=500,
        topk=100,
        old_threshold=None,
        new_threshold=None,
    ):
        """
        Annotate cells using second-layer hierarchical concepts with hierarchical constraints.

        Workflow
        --------
        1. Score second-layer child concepts using standard z-score-based scoring.
        2. Use first-layer scores to determine the best parent concept for each cell.
        3. Restrict child assignment to only the children of that parent concept.

        Parameters
        ----------
        concepts : list[dict], optional
            Flattened second-layer concepts.
            If None, `self.hconcepts` will be used.
        scores_old : np.ndarray, optional
            First-layer concept score matrix (cells × first-layer concepts).
            If None, `self.annotation_scores` will be used.
        dataset : str, optional
            Dataset name. If provided and different from the currently loaded
            dataset, the dataset will be reloaded.
        batch_size : int, default=500
            Batch size used when loading dataset if needed.
        topk : int, default=100
            Keep top-k genes for each second-layer concept before annotation.
        old_threshold : float, optional
            Threshold for first-layer concept score.
        new_threshold : float, optional
            Threshold for second-layer concept score within the selected parent.

        Returns
        -------
        scores_level : np.ndarray
            Cell × second-layer concept score matrix.
        concept_names : list[str]
            Second-layer concept names.
        pred_labels : list[str]
            Hierarchically assigned second-layer concept labels.
        """
        import numpy as np

        if dataset is not None and (self.dataset_handler is None or self.dataset_name != dataset):
            self._load_dataset(dataset_name=dataset, batch_size=batch_size)

        if self.test_data is None or self.gene_names is None:
            raise ValueError("Dataset not loaded. Please provide dataset=... or load data first.")

        if concepts is None:
            if getattr(self, "hconcepts", None) is not None:
                concepts = self.hconcepts
            else:
                raise ValueError(
                    "No hierarchical concepts available. Run hconcept() first or pass concepts explicitly."
                )

        if scores_old is None:
            if getattr(self, "annotation_scores", None) is not None:
                scores_old = self.annotation_scores
            else:
                raise ValueError(
                    "No first-layer concept scores available. "
                    "Run annotation() first or pass scores_old explicitly."
                )

        if getattr(self, "hconcept_parent_ids", None) is None:
            raise ValueError(
                "No parent-child mapping available. Run hconcept() first."
            )

        gene_to_idx = {str(g): i for i, g in enumerate(self.gene_names)}

        # top-k child concepts
        new_concepts = self.generate_topk_concepts(concepts, k=topk)

        # second-layer flat scores
        scores_level, concept_names, _ = self.assign_cells_by_concepts_zscore(
            new_concepts,
            self.test_data,
            gene_to_idx
        )

        # parent and child names
        old_concept_names = None
        if getattr(self, "concepts", None) is not None:
            old_concept_names = [c["name"] for c in self.concepts]

        new_concept_names = [c["name"] for c in new_concepts]

        # hierarchical assignment
        df_assign = self.hierarchical_assign(
            score_old=scores_old,
            score_new=scores_level,
            parent_ids_for_new=self.hconcept_parent_ids,
            old_concept_names=old_concept_names,
            new_concept_names=new_concept_names,
            old_threshold=old_threshold,
            new_threshold=new_threshold,
        )

        pred_labels = df_assign["new_concept_name"].tolist()

        # cache
        self.hconcepts_topk = new_concepts
        self.hannotation_scores = scores_level
        self.hannotation_df = df_assign
        self.pred_hconcepts = pred_labels

        # save
        dataset_dir = self._dataset_result_dir(self.dataset_name)

        self.save_json(
            pred_labels,
            dataset_dir / f"{self.dataset_name}_hierarchical_pred_labels.json"
        )
        df_assign.to_csv(
            dataset_dir / f"{self.dataset_name}_hierarchical_assignments.csv",
            index=False
        )
        np.save(
            dataset_dir / f"{self.dataset_name}_hierarchical_scores.npy",
            scores_level
        )

        return scores_level, concept_names, pred_labels