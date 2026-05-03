"""
models/gnn.py — Graph Neural Network for cross-sectional stock relationships.

Models two types of edges between stocks:
  1. Sector membership (same sector → edge)
  2. Return correlation (Pearson |corr| > threshold → edge)

GNN learns to propagate information along edges, capturing contagion,
sector rotation, and supply-chain effects that univariate models miss.

Architecture: GraphSAGE (Hamilton et al., 2017) — scales to full HOSE/HNX
universe without requiring the full adjacency matrix in memory.
"""
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_settings
from models.base import BaseModel, ModelMetrics

try:
    from torch_geometric.data import Data, DataLoader as GeoDataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Fallback: use a simple MLP if torch-geometric is not installed
    SAGEConv = None

settings = get_settings()
logger = structlog.get_logger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Graph construction ───────────────────────────────────────────────────────
def build_stock_graph(
    features_df: pd.DataFrame,       # index=ticker, columns=features
    sector_map: Dict[str, str],       # {ticker: sector}
    returns_df: pd.DataFrame,         # index=date, columns=ticker (for corr)
    corr_threshold: float = 0.6,
    max_corr_edges: int = 5,          # max correlation neighbours per node
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build edge_index and node feature matrix for the stock universe.

    Returns:
      node_features: (N, F) float32 array
      edge_index:    (2, E) int64 array (COO format)
      ticker_order:  list of tickers in node order
    """
    tickers = list(features_df.index)
    N = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    # ── Node features ────────────────────────────────────────────────────
    node_features = features_df.fillna(0).values.astype(np.float32)

    # ── Sector edges ─────────────────────────────────────────────────────
    edges = set()
    from itertools import combinations
    sector_groups: Dict[str, List[str]] = {}
    for ticker, sector in sector_map.items():
        if ticker in ticker_to_idx and sector:
            sector_groups.setdefault(sector, []).append(ticker)

    for sector, members in sector_groups.items():
        for t1, t2 in combinations(members, 2):
            i, j = ticker_to_idx[t1], ticker_to_idx[t2]
            edges.add((i, j))
            edges.add((j, i))  # undirected → both directions

    # ── Correlation edges ─────────────────────────────────────────────────
    common_tickers = [t for t in tickers if t in returns_df.columns]
    if len(common_tickers) >= 2:
        corr_matrix = returns_df[common_tickers].corr().fillna(0)
        for ticker in common_tickers:
            if ticker not in ticker_to_idx:
                continue
            i = ticker_to_idx[ticker]
            corr_row = corr_matrix[ticker].drop(ticker, errors="ignore")
            top_corr = corr_row[corr_row.abs() > corr_threshold].nlargest(max_corr_edges)
            for neighbour, _ in top_corr.items():
                if neighbour in ticker_to_idx:
                    j = ticker_to_idx[neighbour]
                    edges.add((i, j))
                    edges.add((j, i))

    if edges:
        edge_index = np.array(list(edges), dtype=np.int64).T  # (2, E)
    else:
        # Self-loops only (degenerate case)
        edge_index = np.array([[i, i] for i in range(N)], dtype=np.int64).T

    return node_features, edge_index, tickers


# ─── GNN model ────────────────────────────────────────────────────────────────
class GraphSAGEEncoder(nn.Module):
    """
    3-layer GraphSAGE encoder with residual connections.
    Outputs a fixed-size embedding for each node (stock).
    """

    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            # Fallback: pure MLP (no graph structure)
            self.conv1 = nn.Linear(in_channels, hidden)
            self.conv2 = nn.Linear(hidden, hidden)
            self.conv3 = nn.Linear(hidden, out_channels)
            self._use_sage = False
        else:
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.conv3 = SAGEConv(hidden, out_channels)
            self._use_sage = True

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self._use_sage and edge_index is not None:
            h = F.relu(self.norm1(self.conv1(x, edge_index)))
            h = self.dropout(h)
            h = F.relu(self.norm2(self.conv2(h, edge_index)))
            h = self.dropout(h)
            return self.conv3(h, edge_index)
        else:
            h = F.relu(self.norm1(self.conv1(x)))
            h = self.dropout(h)
            h = F.relu(self.norm2(self.conv2(h)))
            h = self.dropout(h)
            return self.conv3(h)


class GNNRankingHead(nn.Module):
    """Stock scoring head on top of GNN embeddings."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings).squeeze(-1)


class GNNForecaster(BaseModel):
    """
    GNN-based cross-sectional return predictor.
    Uses graph structure (sector + correlation) to improve stock scoring
    beyond what univariate models see.
    """

    MODEL_NAME = "GNN_GraphSAGE"

    def __init__(
        self,
        n_features: int = 30,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 1,    # full-graph training (single batch)
    ):
        super().__init__()
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._encoder: Optional[GraphSAGEEncoder] = None
        self._head: Optional[GNNRankingHead] = None
        self._model_kwargs = dict(
            in_channels=n_features, hidden=hidden_dim,
            out_channels=embedding_dim, dropout=dropout,
        )

    def train(
        self,
        train_df: pd.DataFrame,        # rows=stocks, cols=features + 'fwd_return' + 'ticker'
        sector_map: Dict[str, str],
        returns_df: pd.DataFrame,      # historical returns for corr graph
        val_df: Optional[pd.DataFrame] = None,
        experiment_name: Optional[str] = None,
    ) -> ModelMetrics:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

        feature_cols = [c for c in train_df.columns
                        if c not in ("ticker", "fwd_return", "date")]
        train_df = train_df.set_index("ticker")

        node_features, edge_index, ticker_order = build_stock_graph(
            features_df=train_df[feature_cols],
            sector_map=sector_map,
            returns_df=returns_df,
        )

        X = torch.tensor(node_features, dtype=torch.float32).to(DEVICE)
        E = torch.tensor(edge_index, dtype=torch.long).to(DEVICE) if edge_index.size > 0 else None
        y = torch.tensor(
            [train_df.loc[t, "fwd_return"] if t in train_df.index else 0.0
             for t in ticker_order],
            dtype=torch.float32,
        ).to(DEVICE)

        self._encoder = GraphSAGEEncoder(**self._model_kwargs).to(DEVICE)
        self._head = GNNRankingHead(self._model_kwargs["out_channels"]).to(DEVICE)
        optimizer = optim.Adam(
            list(self._encoder.parameters()) + list(self._head.parameters()),
            lr=self.learning_rate, weight_decay=1e-4,
        )

        import torch.optim as optim
        criterion = nn.MSELoss()

        with mlflow.start_run(run_name=f"GNN_{datetime.now():%Y%m%d_%H%M}") as run:
            mlflow.log_params({
                "model": self.MODEL_NAME,
                "n_nodes": len(ticker_order),
                "n_edges": edge_index.shape[1] if len(edge_index.shape) > 1 else 0,
                **self._model_kwargs,
            })

            for epoch in range(self.epochs):
                self._encoder.train()
                self._head.train()
                optimizer.zero_grad()
                embeddings = self._encoder(X, E)
                scores = self._head(embeddings)
                loss = criterion(scores, y)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._encoder.parameters()) + list(self._head.parameters()), 1.0
                )
                optimizer.step()

                if epoch % 20 == 0:
                    mlflow.log_metric("train_loss", float(loss), step=epoch)

            # Eval: information coefficient
            self._encoder.eval()
            self._head.eval()
            with torch.no_grad():
                embeddings = self._encoder(X, E)
                scores = self._head(embeddings).cpu().numpy()
            ic = float(pd.Series(scores).corr(pd.Series(y.cpu().numpy())))
            mlflow.log_metric("train_ic", ic)

            # Save
            with tempfile.TemporaryDirectory() as tmp:
                torch.save({
                    "encoder": self._encoder.state_dict(),
                    "head": self._head.state_dict(),
                    "ticker_order": ticker_order,
                    "model_kwargs": self._model_kwargs,
                }, os.path.join(tmp, "gnn.pt"))
                mlflow.log_artifacts(tmp, artifact_path="model")

            self._run_id = run.info.run_id
            return ModelMetrics(
                model_name=self.MODEL_NAME,
                run_id=run.info.run_id,
                extra={"train_ic": ic},
            )

    def predict(self, df: pd.DataFrame, sector_map: Dict[str, str] = None,
                returns_df: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        if self._encoder is None:
            raise RuntimeError("Model not loaded")

        feature_cols = [c for c in df.columns
                        if c not in ("ticker", "fwd_return", "date")]
        df_idx = df.set_index("ticker") if "ticker" in df.columns else df
        node_features = df_idx[feature_cols].fillna(0).values.astype(np.float32)
        tickers = list(df_idx.index)

        X = torch.tensor(node_features).to(DEVICE)
        edge_index = None
        if sector_map and returns_df is not None:
            _, ei, _ = build_stock_graph(df_idx[feature_cols], sector_map, returns_df)
            edge_index = torch.tensor(ei, dtype=torch.long).to(DEVICE)

        self._encoder.eval()
        self._head.eval()
        with torch.no_grad():
            embeddings = self._encoder(X, edge_index)
            raw_scores = self._head(embeddings).cpu().numpy()

        result = pd.DataFrame({"ticker": tickers, "gnn_score": raw_scores})
        # Normalise 0-100
        mn, mx = result["gnn_score"].min(), result["gnn_score"].max()
        result["gnn_score"] = (result["gnn_score"] - mn) / max(mx - mn, 1e-8) * 100
        return result

    @classmethod
    def load(cls, mlflow_run_id: str) -> "GNNForecaster":
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        artifact_uri = mlflow.get_run(mlflow_run_id).info.artifact_uri
        instance = cls()
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                f"{artifact_uri}/model/gnn.pt", dst_path=tmp
            )
            checkpoint = torch.load(local, map_location=DEVICE)
        instance._model_kwargs = checkpoint["model_kwargs"]
        instance._encoder = GraphSAGEEncoder(**instance._model_kwargs).to(DEVICE)
        instance._encoder.load_state_dict(checkpoint["encoder"])
        instance._head = GNNRankingHead(instance._model_kwargs["out_channels"]).to(DEVICE)
        instance._head.load_state_dict(checkpoint["head"])
        instance._run_id = mlflow_run_id
        return instance
