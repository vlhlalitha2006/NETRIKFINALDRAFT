from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


DEFAULT_TRAIN_PATH = Path("data/raw/TRAIN.csv")
DEFAULT_TEST_PATH = Path("data/raw/TEST.csv")
DEFAULT_OUTPUT_DIR = Path("data/graph")
DEFAULT_K = 10


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def extract_node_ids(dataframe: pd.DataFrame, id_column: str = "Loan_ID") -> np.ndarray:
    if id_column in dataframe.columns:
        node_ids = dataframe[id_column].astype(str).to_numpy()
    else:
        node_ids = dataframe.index.astype(str).to_numpy()
    return node_ids


def build_numeric_feature_matrix(
    dataframe: pd.DataFrame,
    target_column: str = "Loan_Status",
    id_column: str = "Loan_ID",
) -> np.ndarray:
    working_df = dataframe.copy()
    drop_columns: list[str] = []
    if target_column in working_df.columns:
        drop_columns.append(target_column)
    if id_column in working_df.columns:
        drop_columns.append(id_column)
    if drop_columns:
        working_df = working_df.drop(columns=drop_columns)

    numeric_columns = working_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric columns found for similarity graph construction.")

    numeric_df = working_df[numeric_columns]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("l2_norm", Normalizer(norm="l2")),
        ]
    )
    features = numeric_pipeline.fit_transform(numeric_df)
    return np.asarray(features, dtype=np.float32)


def build_knn_edge_index(features: np.ndarray, k: int) -> np.ndarray:
    num_nodes = int(features.shape[0])
    if num_nodes < 2:
        raise ValueError("Need at least 2 nodes to build a similarity graph.")

    effective_k = min(k, num_nodes - 1)
    neighbors_to_query = effective_k + 1

    nn_model = NearestNeighbors(
        n_neighbors=neighbors_to_query,
        metric="cosine",
        algorithm="auto",
    )
    nn_model.fit(features)
    _, neighbor_indices = nn_model.kneighbors(features, return_distance=True)

    edge_set: set[tuple[int, int]] = set()
    for src_idx in range(num_nodes):
        neighbor_row = neighbor_indices[src_idx]
        added = 0
        for dst_idx in neighbor_row:
            if int(dst_idx) == src_idx:
                continue
            edge_set.add((src_idx, int(dst_idx)))
            edge_set.add((int(dst_idx), src_idx))
            added += 1
            if added >= effective_k:
                break

    sorted_edges = np.array(sorted(edge_set), dtype=np.int64)
    edge_index = sorted_edges.T
    return edge_index


def estimate_memory_mb(features: np.ndarray, edge_index: np.ndarray, k: int, num_nodes: int) -> float:
    graph_bytes = features.nbytes + edge_index.nbytes
    knn_runtime_bytes = num_nodes * (k + 1) * (8 + 8)
    total_mb = (graph_bytes + knn_runtime_bytes) / (1024 * 1024)
    return float(total_mb)


def save_graph_artifacts(
    output_dir: Path,
    edge_index: np.ndarray,
    node_ids: np.ndarray,
    node_features: np.ndarray,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edge_index.npy"
    node_ids_path = output_dir / "node_ids.npy"
    node_features_path = output_dir / "node_features.npy"

    np.save(edge_path, edge_index.astype(np.int64))
    np.save(node_ids_path, node_ids)
    np.save(node_features_path, node_features.astype(np.float32))
    return edge_path, node_ids_path, node_features_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline applicant similarity graph.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--target-column", type=str, default="Loan_Status")
    parser.add_argument("--id-column", type=str, default="Loan_ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("k must be > 0.")
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {args.train_csv}")
    if not args.test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {args.test_csv}")

    train_df = load_dataframe(args.train_csv)
    test_df = load_dataframe(args.test_csv)
    dataframe = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    node_ids = extract_node_ids(dataframe, id_column=args.id_column)
    node_features = build_numeric_feature_matrix(
        dataframe,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    edge_index = build_knn_edge_index(node_features, k=args.k)

    edge_path, node_ids_path, node_features_path = save_graph_artifacts(
        output_dir=args.output_dir,
        edge_index=edge_index,
        node_ids=node_ids,
        node_features=node_features,
    )

    num_nodes = int(node_features.shape[0])
    num_edges = int(edge_index.shape[1])
    avg_degree = float(num_edges / max(num_nodes, 1))
    memory_mb = estimate_memory_mb(
        features=node_features,
        edge_index=edge_index,
        k=min(args.k, max(num_nodes - 1, 1)),
        num_nodes=num_nodes,
    )

    print(f"Saved edge index: {edge_path}")
    print(f"Saved node ids: {node_ids_path}")
    print(f"Saved node features: {node_features_path}")
    print(f"total_nodes: {num_nodes}")
    print(f"total_edges: {num_edges}")
    print(f"avg_degree: {avg_degree:.2f}")
    print(f"memory_estimate_mb: {memory_mb:.2f}")


if __name__ == "__main__":
    main()
