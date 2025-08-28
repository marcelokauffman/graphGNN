#!/usr/bin/env python3
"""
SHEPHERD GNN training CLI for GPU VM (Runpod-ready).

Implements:
- Data loader that safely loads .pt dicts saved with pandas DataFrame (PyTorch 2.6 weights_only change)
- Falls back to NumPy artifacts if torch.load fails due to pandas version mismatches
- GAT encoder + DistMult decoder with tanh
- Hinge ranking loss with per-relation negative sampling (edge batching)
- Checkpointing, metrics logging, and embeddings export

Usage:
  python train_gnn.py --data-dir ./data/graphNN --output-dir ./outputs --epochs 100
"""

from __future__ import annotations
import argparse
import json
import time
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np


def _safe_torch_load(path: Path):
    """Load torch .pt that may contain pandas DataFrame under PyTorch>=2.6.
    Prefers safe_globals, and falls back to weights_only=False for trusted files.
    """
    from torch.serialization import add_safe_globals
    try:
        add_safe_globals([pd.DataFrame])
    except Exception:
        pass
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        # As per error hint; only do this if file is trusted (here it is ours)
        return torch.load(path, map_location='cpu', weights_only=False)


class VMKGDataLoader:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "conversion_summary.json", "r") as f:
            self.summary = json.load(f)
        # mappings.pkl is a Python pickle
        import pickle as _pkl
        with open(self.data_dir / "mappings.pkl", "rb") as f:
            self.mappings = _pkl.load(f)

    def load_split(self, split: str) -> Data:
        pt_path = self.data_dir / f"{split}_pyg_data.pt"
        try:
            data_dict = _safe_torch_load(pt_path)
            data = Data(
                x=data_dict['x'].float(),
                edge_index=data_dict['edge_index'].long(),
                edge_attr=data_dict['edge_attr'].long(),
            )
            data.num_nodes = int(data_dict['num_nodes'])
            data.num_edges = int(data_dict['num_edges'])
            data.num_node_features = int(data_dict['num_node_features'])
            data.num_relation_types = int(data_dict['num_relation_types'])
            return data
        except Exception as e:
            print(f"⚠️ Failed to load {pt_path} via torch.load due to: {e}\n   Falling back to NumPy files (edges/features).")
            # Fallback: rebuild from numpy artifacts
            edges_path = self.data_dir / f"{split}_edges.npy"
            x_path = self.data_dir / "node_features.npy"
            if not edges_path.exists() or not x_path.exists():
                raise FileNotFoundError(
                    f"Missing fallback files: {edges_path} and/or {x_path}."
                )
            edges = np.load(edges_path)  # shape (E, 3): src, dst, rel_idx
            if edges.ndim != 2 or edges.shape[1] < 3:
                raise ValueError(f"Unexpected edges shape: {edges.shape}")
            x = np.load(x_path).astype(np.float32)
            edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)
            edge_attr = torch.tensor(edges[:, 2].copy(), dtype=torch.long)
            data = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            data.num_nodes = int(x.shape[0])
            data.num_edges = int(edges.shape[0])
            data.num_node_features = int(x.shape[1])
            data.num_relation_types = int(self.summary.get('relation_types', len(self.mappings.get('relation_to_idx', {}))))
            return data


class ShepherdGAT(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_relations: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_attention_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_edge_attr: bool = True):
        super().__init__()
        if num_nodes is None:
            raise ValueError("num_nodes is required")
        if use_edge_attr and num_relations is None:
            raise ValueError("num_relations is required when use_edge_attr=True")

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr

        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)
        if use_edge_attr:
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        gat_input_dim = embedding_dim
        self.input_projection = nn.Linear(gat_input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_attention_heads,
                            heads=num_attention_heads, dropout=dropout,
                            add_self_loops=True, concat=True)
                )
            elif i == num_layers - 1:
                self.gat_layers.append(
                    GATConv(hidden_dim, embedding_dim,
                            heads=1, dropout=dropout,
                            add_self_loops=True, concat=False)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_attention_heads,
                            heads=num_attention_heads, dropout=dropout,
                            add_self_loops=True, concat=True)
                )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers-1)])
        self.layer_norms.append(nn.LayerNorm(embedding_dim))
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        if self.use_edge_attr:
            nn.init.xavier_uniform_(self.relation_embedding.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        node_idx = torch.arange(x.size(0), device=x.device)
        h = self.node_embedding(node_idx)
        h = F.relu(self.input_projection(h))
        h = self.dropout(h)
        for i, gat in enumerate(self.gat_layers):
            h_new = gat(h, edge_index)
            if h_new.shape == h.shape:
                h_new = h + h_new
            h = h_new
            if i < len(self.gat_layers) - 1:
                h = self.layer_norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                h = self.layer_norms[i](h)
        return h


def create_negative_triples(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int, num_neg: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    E = edge_index.size(1)
    if num_neg is None:
        num_neg = E
    device = edge_index.device
    idx = torch.randint(0, E, (num_neg,), device=device)
    src = edge_index[0, idx].clone()
    dst = edge_index[1, idx].clone()
    rel = edge_attr[idx].clone()
    flip = torch.rand(num_neg, device=device) < 0.5
    rand_nodes = torch.randint(0, num_nodes, (num_neg,), device=device)
    src[flip] = rand_nodes[flip]
    dst[~flip] = rand_nodes[~flip]
    return torch.stack([src, dst], dim=0), rel


def distmult_scores(node_emb: torch.Tensor, rel_emb_table: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    x_u = node_emb[edge_index[0]]
    x_v = node_emb[edge_index[1]]
    r = rel_emb_table[edge_attr]
    return torch.tanh((x_u * r * x_v).sum(dim=1))


def link_prediction_hinge_loss(
    model: ShepherdGAT,
    data: Data,
    margin: float = 1.0,
    num_neg: int | None = None,
    edge_batch_size: int = 200_000,
    metric_max_edges: int = 500_000,
    neg_mult: float = 1.0,
):
    """Compute hinge loss in edge batches to keep memory bounded.

    Also computes AUC/AP on up to metric_max_edges edges for speed.
    """
    node_embeddings = model(data)

    E = data.edge_index.size(1)
    if num_neg is None:
        num_neg = min(E, edge_batch_size)

    total_loss = 0.0
    n_batches = 0
    metric_pos = []
    metric_neg = []

    for start in range(0, E, edge_batch_size):
        end = min(E, start + edge_batch_size)
        pos_idx = data.edge_index[:, start:end]
        pos_rel = data.edge_attr[start:end]
        # Use integer negatives-per-positive (k)
        P = end - start
        k = max(1, int(round(float(neg_mult))))
        n_neg = P * k
        neg_idx, neg_rel = create_negative_triples(pos_idx, pos_rel, data.num_nodes, num_neg=n_neg)

        pos_scores = distmult_scores(node_embeddings, model.relation_embedding.weight, pos_idx, pos_rel)  # [P]
        neg_scores = distmult_scores(node_embeddings, model.relation_embedding.weight, neg_idx, neg_rel)  # [P*k]
        # Reshape negatives to [P, k] and compute hinge per (pos, neg)
        neg_scores = neg_scores.view(P, k)
        batch_loss = F.relu(margin - pos_scores.unsqueeze(1) + neg_scores).mean()
        total_loss += batch_loss
        n_batches += 1

        # For metrics, repeat positives to match negatives count
        metric_pos.append(pos_scores.detach().repeat_interleave(k))  # [P*k]
        metric_neg.append(neg_scores.detach().reshape(-1))           # [P*k]

    loss = total_loss / max(1, n_batches)

    with torch.no_grad():
        if metric_pos:
            pos_scores = torch.cat(metric_pos)
            neg_scores = torch.cat(metric_neg)
            pos_scores = pos_scores[:metric_max_edges]
            neg_scores = neg_scores[:metric_max_edges]
            pos_p = (pos_scores + 1) / 2
            neg_p = (neg_scores + 1) / 2
            probs = torch.cat([pos_p, neg_p]).clamp(0, 1)
            labels = torch.cat([torch.ones_like(pos_p), torch.zeros_like(neg_p)])
            auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
            ap = average_precision_score(labels.cpu().numpy(), probs.cpu().numpy())
        else:
            auc = ap = 0.0

    return loss, float(auc), float(ap)


@torch.no_grad()
def evaluate(model: ShepherdGAT, data: Data, margin: float, edge_batch_size: int = 200_000, metric_max_edges: int = 500_000, neg_mult: float = 1.0):
    model.eval()
    loss, auc, ap = link_prediction_hinge_loss(
        model, data, margin, edge_batch_size=edge_batch_size, metric_max_edges=metric_max_edges, neg_mult=neg_mult
    )
    return float(loss.item()), float(auc), float(ap)


def _ensure_cuda_alloc_conf():
    """Ensure PYTORCH_CUDA_ALLOC_CONF has a valid format to avoid parse errors.
    If unset or malformed, set a safe default. Also normalize boolean case.
    """
    key = 'PYTORCH_CUDA_ALLOC_CONF'
    val = os.environ.get(key)
    if not val or ':' not in val:
        os.environ[key] = 'expandable_segments:true'
    else:
        os.environ[key] = val.replace('True', 'true').replace('False', 'false')


def train(args):
    # Sanitize allocator config before any CUDA init
    _ensure_cuda_alloc_conf()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outdir = Path(args.output_dir)
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (outdir / 'embeddings').mkdir(parents=True, exist_ok=True)
    (outdir / 'results').mkdir(parents=True, exist_ok=True)

    loader = VMKGDataLoader(args.data_dir)
    train_data = loader.load_split('train').to(device)
    val_data = loader.load_split('val').to(device)
    test_data = loader.load_split('test').to(device)

    model = ShepherdGAT(
        num_nodes=train_data.num_nodes,
        num_relations=train_data.num_relation_types,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_attention_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_edge_attr=True,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    patience = 0
    start = time.time()
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    metrics_log = outdir / 'results' / 'metrics.log'
    with open(metrics_log, 'w') as f:
        f.write('epoch,train_loss,val_loss,train_auc,val_auc,lr\n')

    # Autocast + scaler setup
    if args.amp != 'off' and torch.cuda.is_available():
        if args.amp == 'bf16' and torch.cuda.is_bf16_supported():
            amp_cm = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            scaler = None
        elif args.amp == 'fp16':
            amp_cm = torch.cuda.amp.autocast(dtype=torch.float16)
            scaler = torch.cuda.amp.GradScaler()
        else:
            amp_cm = nullcontext()
            scaler = None
    else:
        amp_cm = nullcontext()
        scaler = None

    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad()
        with amp_cm:
            loss, auc, ap = link_prediction_hinge_loss(
                model, train_data, args.margin, edge_batch_size=args.edge_batch_size, metric_max_edges=args.metric_max_edges, neg_mult=args.neg_mult
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        with amp_cm:
            vloss, vauc, vap = evaluate(
                model, val_data, args.margin, edge_batch_size=args.edge_batch_size, metric_max_edges=args.metric_max_edges, neg_mult=args.neg_mult
            )
        sched.step(vloss)

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(vloss))
        history["train_auc"].append(float(auc))
        history["val_auc"].append(float(vauc))

        if (epoch % 5 == 0) or (epoch < 10):
            print(f"Epoch {epoch:03d} | train loss {loss:.4f} | val loss {vloss:.4f} | train AUC {auc:.4f} | val AUC {vauc:.4f}")
        with open(metrics_log, 'a') as f:
            lr_cur = float(optim.param_groups[0]['lr'])
            f.write(f"{epoch},{float(loss.item()):.6f},{float(vloss):.6f},{float(auc):.6f},{float(vauc):.6f},{lr_cur:.6e}\n")

        if vloss < best_val - 1e-4:
            best_val = vloss
            patience = 0
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_loss': best_val,
                'config': vars(args),
            }
            torch.save(ckpt, outdir / 'checkpoints' / 'best_model.pt')
        else:
            patience += 1
            if patience >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {patience})")
                break

    dur = (time.time() - start) / 60
    print(f"Training finished in {dur:.1f} min | best val loss {best_val:.4f}")

    with open(outdir / 'results' / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    with amp_cm:
        tloss, tauc, tap = evaluate(
            model, test_data, args.margin, edge_batch_size=args.edge_batch_size, metric_max_edges=args.metric_max_edges, neg_mult=args.neg_mult
        )
    final = {"loss": tloss, "auc": tauc, "average_precision": tap}
    print(f"Test: loss {tloss:.4f} | AUC {tauc:.4f} | AP {tap:.4f}")
    with open(outdir / 'results' / 'final_results.json', 'w') as f:
        json.dump({"test_metrics": final, "config": vars(args)}, f, indent=2)

    model.eval()
    with torch.no_grad():
        emb = model(train_data)
    torch.save(emb.cpu(), outdir / 'embeddings' / 'node_embeddings.pt')


def build_parser():
    p = argparse.ArgumentParser(description="SHEPHERD GNN training (VM)")
    p.add_argument('--data-dir', required=True, help='Path with *_pyg_data.pt and/or *_edges.npy + node_features.npy')
    p.add_argument('--output-dir', default='./outputs', help='Where to write checkpoints/results')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--margin', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--embedding-dim', type=int, default=256)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--num-heads', type=int, default=8)
    p.add_argument('--num-layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--neg-mult', type=float, default=1.0, help='Negatives per positive ratio in each edge batch')
    p.add_argument('--edge-batch-size', type=int, default=200_000, help='Edges per batch for loss computation')
    p.add_argument('--metric-max-edges', type=int, default=500_000, help='Max edges sampled for AUC/AP')
    p.add_argument('--amp', type=str, choices=['off','bf16','fp16'], default='bf16', help='Automatic mixed precision for memory/speed (bf16 preferred on Ampere+)')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    train(args)
