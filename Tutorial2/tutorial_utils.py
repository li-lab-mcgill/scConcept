"""Input and scoring helpers for tutorials that start from saved scConcept outputs."""
from pathlib import Path
import json
import os
import numpy as np
from scipy import sparse
from scipy.io import loadmat
import torch
from scConcept import ScConcept
from ECRTM.singlecell_dataset import SingleCellDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = PROJECT_ROOT / "Tutorial2"
DATA_DIR = PROJECT_ROOT / "Datasets"
_CONFIG_PATH = TUTORIAL_DIR / "Inputs/data_paths.local.json"
_INPUT_PATHS = {}


def configure_inputs(data_dir=None, config_path=None):
    """Refresh paths when a notebook setup cell is rerun after configuration edits."""
    global DATA_DIR, _CONFIG_PATH, _INPUT_PATHS
    DATA_DIR = Path(data_dir or os.environ.get("SCCONCEPT_DATA_DIR", PROJECT_ROOT / "Datasets")).expanduser()
    _CONFIG_PATH = Path(config_path or os.environ.get("SCCONCEPT_INPUT_CONFIG", TUTORIAL_DIR / "Inputs/data_paths.local.json")).expanduser()
    _INPUT_PATHS = json.loads(_CONFIG_PATH.read_text()) if _CONFIG_PATH.is_file() else {}
    if not isinstance(_INPUT_PATHS, dict):
        raise ValueError(f"Expected a JSON object mapping input keys to paths: {_CONFIG_PATH}")
    return DATA_DIR


configure_inputs()


def input_path(key, relative_path, *aliases):
    """Resolve an input from the optional path map or the repository Datasets directory."""
    if key in _INPUT_PATHS:
        value = Path(_INPUT_PATHS[key]).expanduser()
        return value if value.is_absolute() else DATA_DIR / value
    candidates = [DATA_DIR / relative_path, *(DATA_DIR / name for name in aliases)]
    return next((p for p in candidates if p.exists()), candidates[0])


def dataset_path(name, *aliases):
    """Locate an already processed .mat dataset; do not preprocess it again."""
    return input_path(f"mat.{name}", f"{name}.mat", *(f"{a}.mat" for a in aliases))


def matlab_string(value):
    """Decode nested MATLAB strings without changing gene or cell-label spelling."""
    while isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return "".join(matlab_string(v) for v in value.reshape(-1))
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_mat_test(path, dtype=None):
    """Read the package .mat schema, keeping large sparse test matrices sparse."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}. Configure its path before running the analysis.")
    data = loadmat(path, variable_names=["bow_test", "voc", "label_test"])
    missing = {"bow_test", "voc"} - data.keys()
    if missing:
        raise ValueError(f"Missing .mat fields in {path}: {sorted(missing)}")
    expression = data["bow_test"]
    if dtype is not None:
        expression = expression.astype(dtype)
    expression = expression.tocsr() if sparse.issparse(expression) else np.asarray(expression)
    genes = [matlab_string(v) for v in data["voc"].reshape(-1)]
    labels = None
    if "label_test" in data:
        labels = np.asarray([matlab_string(v) for v in data["label_test"].reshape(-1)])
    if expression.shape[1] != len(genes) or len(set(genes)) != len(genes):
        raise ValueError(f"Gene names do not match expression columns: {path}")
    if labels is not None and expression.shape[0] != len(labels):
        raise ValueError(f"Labels do not match expression rows: {path}")
    values = expression.data if sparse.issparse(expression) else expression
    if not np.isfinite(values).all():
        raise ValueError(f"Expression contains non-finite values: {path}")
    return expression, genes, labels


def load_api_dataset(path, batch_size=512):
    """Use the same CPU SingleCellDataset loader as Tutorial 1 and ScConcept."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return SingleCellDataset(dataset_name=path.stem, batch_size=batch_size, data_dir=str(path.parent))


def load_saved_concepts(path):
    """Accept a saved concept list or the {'concepts': [...]} generation response."""
    obj = ScConcept.load_json(path)
    concepts = obj["concepts"] if isinstance(obj, dict) and "concepts" in obj else obj
    if not isinstance(concepts, list) or not concepts:
        raise ValueError(f"Expected a nonempty concept list: {path}")
    for concept in concepts:
        if not isinstance(concept.get("name"), str) or not isinstance(concept.get("genes"), list):
            raise ValueError(f"Each concept requires a name and an ordered gene list: {path}")
    return concepts


def score_saved_concepts(concepts, expression, genes, top_k=100, require_matches=True):
    """Apply the package z-score annotation rule to one gene block at a time.

    Gene blocks avoid densifying an entire atlas. Each block still contains all
    cells, preserving full-cohort standardization and listed marker order.
    """
    index = {g: i for i, g in enumerate(genes)}
    definitions = ScConcept.generate_topk_concepts(concepts, k=top_k)
    scores = np.zeros((expression.shape[0], len(definitions)), dtype=float)
    matches = []
    for i, concept in enumerate(definitions):
        selected = concept["genes"]
        columns = [index[g] for g in selected if g in index]
        if columns:
            block = expression[:, columns]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            # Local names preserve duplicate marker entries as separate columns.
            local_names = [str(j) for j in range(len(columns))]
            local = [{"name": concept["name"], "genes": local_names}]
            tensor = torch.as_tensor(np.ascontiguousarray(block))
            values, _, _ = ScConcept.assign_cells_by_concepts_zscore(local, tensor, dict(zip(local_names, range(len(columns)))))
            scores[:, i] = values[:, 0]
        elif require_matches:
            raise ValueError(f"No input genes match concept {concept['name']!r}.")
        matches.append(dict(concept=concept["name"], n_requested=len(selected), n_matched=len(columns),
                            missing_genes=[g for g in selected if g not in index]))
    names = [c["name"] for c in definitions]
    return scores, names, matches


def runtime_paths():
    """Record the package source and input mapping with analysis outputs."""
    import hashlib
    import inspect
    api_file = Path(inspect.getfile(ScConcept)).resolve()
    return dict(project_root=str(PROJECT_ROOT), data_dir=str(DATA_DIR),
                api_source=str(api_file), api_sha256=hashlib.sha256(api_file.read_bytes()).hexdigest(),
                input_config=str(_CONFIG_PATH) if _CONFIG_PATH.is_file() else None,
                input_config_sha256=hashlib.sha256(_CONFIG_PATH.read_bytes()).hexdigest() if _CONFIG_PATH.is_file() else None)


# Paper TC/TD preserve the reported background and denominator definitions.
from collections import Counter

compute_coherence_from_gene_lists = ScConcept.compute_coherence_from_gene_lists

def TD_eva_topics(topics):
    """Count genes occurring exactly once using the paper's K-times-T denominator."""
    K = len(topics)
    T = len(topics[0])
    all_genes = []
    for topic in topics:
        all_genes.extend(topic)
    gene_freq = Counter(all_genes)
    unique_genes_count = sum((1 for count in gene_freq.values() if count == 1))
    TD = unique_genes_count / (K * T)
    return TD


def read_pathways(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    pathways = []
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                pathways.append([g.upper() for g in fields[2:]])
    if not pathways:
        raise ValueError("The GMT file contains no gene sets.")
    return pathways


def interpretability(concepts, genes, pathways, top_k=10):
    gene_lists = [c["genes"][:top_k] for c in concepts]

    corpus_genes = {g for pathway in pathways for g in pathway}
    queried = {g for values in gene_lists for g in values}
    background_genes = sorted(queried & corpus_genes & set(genes))
    background_idx = {g: i for i, g in enumerate(background_genes)}
    background = np.zeros((len(pathways), len(background_genes)), dtype=np.float64)
    for i, pathway in enumerate(pathways):
        indices = [background_idx[g] for g in pathway if g in background_idx]
        background[i, indices] = 1
    translation = {i: background_idx[g] for i, g in enumerate(genes) if g in background_idx}
    tc = compute_coherence_from_gene_lists(background, genes, gene_lists, translation)
    td = TD_eva_topics(gene_lists)
    flat = [g for values in gene_lists for g in values]
    return dict(TC=float(tc), TD=float(td), TD_unique_fraction=len(set(flat)) / len(flat))


