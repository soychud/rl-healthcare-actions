"""Scalable inference engine: batch-score millions of patients with ensemble confidence."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from src.rl.train import MLP, auto_device
from src.config import N_ACTIONS, ACTION_BUNDLES, SAFETY_CONSTRAINTS


def _load_q_v(state_dim, n_actions, model_dir, prefix, seed, device):
    from src.rl.train import LSTMModel
    model_dir = Path(model_dir)
    q_path = model_dir / f"{prefix}_q_seed{seed}.pt"
    v_path = model_dir / f"{prefix}_v_seed{seed}.pt"
    if not q_path.exists() or not v_path.exists():
        return None, None

    chk = torch.load(str(q_path), map_location=device, weights_only=True)
    has_lstm = any("lstm" in k for k in chk.keys())
    if has_lstm:
        q = LSTMModel(state_dim, 256, n_actions).to(device)
        v = LSTMModel(state_dim, 256, 1).to(device)
    else:
        n_out_q = chk[list(chk.keys())[-1]].shape[0]
        in_w = chk[list(chk.keys())[0]].shape[1] if list(chk.keys())[0].endswith("weight") else state_dim
        q = MLP(in_w, 256, n_out_q).to(device)
        v = MLP(in_w, 256, 1).to(device)
    q.load_state_dict(chk, strict=False)
    v.load_state_dict(torch.load(str(v_path), map_location=device, weights_only=True), strict=False)
    q.eval()
    v.eval()
    return q, v


class InferenceEnsemble:
    def __init__(
        self,
        state_dim: int,
        model_dir: str = "data/models",
        seeds: Optional[List[int]] = None,
        n_actions: int = N_ACTIONS,
        device: Optional[str] = None,
        prefix: str = "iql",
    ):
        self.device = torch.device(device or auto_device())
        self.n_actions = n_actions
        self.state_dim = state_dim
        seeds = seeds or [0, 1, 2, 3, 42]
        self.seeds = seeds
        model_dir = Path(model_dir)

        self.q_nets = []
        self.v_nets = []
        for s in seeds:
            q, v = _load_q_v(state_dim, n_actions, model_dir, prefix, s, self.device)
            if q is not None:
                self.q_nets.append(q)
                self.v_nets.append(v)
        assert len(self.q_nets) > 0, f"No model checkpoints in {model_dir} for seeds {seeds}"
        self.n_ensemble = len(self.q_nets)
        print(f"  Loaded {self.n_ensemble} ensemble members from {model_dir}")

    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> Dict[str, np.ndarray]:
        states = states.to(self.device)
        batch_size = states.shape[0]
        all_q = torch.zeros(self.n_ensemble, batch_size, self.n_actions, device=self.device)
        all_v = torch.zeros(self.n_ensemble, batch_size, 1, device=self.device)
        for i in range(self.n_ensemble):
            all_q[i] = self.q_nets[i](states)
            all_v[i] = self.v_nets[i](states)

        q_mean = all_q.mean(dim=0)
        q_std = all_q.std(dim=0)
        v_mean = all_v.mean(dim=0)
        adv = q_mean - v_mean
        pi = F.softmax(adv / 0.1, dim=-1)

        q_sorted = all_q.sort(dim=0).values
        n = self.n_ensemble
        lo_idx = max(0, int(n * 0.025))
        hi_idx = min(n - 1, int(n * 0.975))
        q_ci_lower = q_sorted[lo_idx]
        q_ci_upper = q_sorted[hi_idx]

        ci_width = (q_ci_upper - q_ci_lower) / 2.0
        conf = 1.0 - torch.clamp(ci_width / (q_mean.abs() + 1e-8), 0, 1)
        conf = conf / (conf.sum(dim=-1, keepdim=True) + 1e-8)

        return {
            "pi": pi.cpu().numpy(),
            "q_values": q_mean.cpu().numpy(),
            "q_std": q_std.cpu().numpy(),
            "v_values": v_mean.cpu().numpy(),
            "advantages": adv.cpu().numpy(),
            "confidence": conf.cpu().numpy(),
            "q_ci_lower": q_ci_lower.cpu().numpy(),
            "q_ci_upper": q_ci_upper.cpu().numpy(),
        }

    def predict_safe(
        self, states: torch.Tensor, safety_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        result = self.predict(states)
        pi = result["pi"]
        if safety_mask is not None:
            pi = pi * safety_mask.astype(np.float32)
            row_sums = pi.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            pi = pi / row_sums
            result["pi"] = pi
        return result


def build_safety_mask(
    n_rows: int,
    n_actions: int,
    labs: Optional[np.ndarray] = None,
    lab_names: Optional[List[str]] = None,
    diagnoses_hadm: Optional[set] = None,
    hadm_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not SAFETY_CONSTRAINTS:
        return np.ones((n_rows, n_actions), dtype=bool)

    mask = np.ones((n_rows, n_actions), dtype=bool)

    if lab_names is not None and labs is not None:
        lab_index = {name: i for i, name in enumerate(lab_names)}
        for c in SAFETY_CONSTRAINTS:
            aid = c.get("action")
            lab = c.get("lab")
            thresh = c.get("threshold")
            direction = c.get("direction")
            if aid is None or lab is None or thresh is None or direction is None:
                continue
            if lab not in lab_index:
                continue
            col_idx = lab_index[lab]
            vals = labs[:, col_idx]
            finite_vals = np.where(np.isfinite(vals), vals, thresh + 999)
            if direction == "below":
                unsafe = finite_vals >= thresh
            elif direction == "above":
                unsafe = finite_vals <= thresh
            else:
                continue
            mask[unsafe, aid] = False

    if diagnoses_hadm is not None and hadm_ids is not None and 5 < n_actions:
        if any(c.get("id") == "S1" for c in SAFETY_CONSTRAINTS):
            for i, hid in enumerate(hadm_ids):
                if hid in diagnoses_hadm:
                    mask[i, 5] = False

    return mask


def batch_inference(
    state_vectors: np.ndarray,
    ensemble: InferenceEnsemble,
    batch_size: int = 4096,
    safety_mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    n = state_vectors.shape[0]
    keys = ["pi", "q_values", "q_std", "v_values", "advantages", "confidence"]
    accum: Dict[str, list] = {k: [] for k in keys}
    n_actions = ensemble.n_actions

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(state_vectors[start:end].astype(np.float32))
        sm = safety_mask[start:end] if safety_mask is not None else None
        result = ensemble.predict_safe(batch, safety_mask=sm)
        for k in keys:
            accum[k].append(result[k])
        if verbose and (start // batch_size) % 50 == 0:
            print(f"  Inference: {end}/{n} ({100*end//n}%)", flush=True)

    stacked = {}
    for k in keys:
        stacked[k] = np.concatenate(accum[k], axis=0)
    stacked["recommended_action"] = stacked["pi"].argmax(axis=1)
    stacked["confidence_score"] = stacked["pi"].max(axis=1)
    return stacked


def export_results(
    results: Dict[str, np.ndarray],
    output_path: str,
    patient_ids: Optional[np.ndarray] = None,
    action_names: Optional[Dict[int, str]] = None,
    format: str = "parquet",
):
    import polars as pl

    anames = action_names or {k: v["name"] for k, v in ACTION_BUNDLES.items()}
    n = results["pi"].shape[0]
    n_actions = results["pi"].shape[1]

    data = {}
    if patient_ids is not None:
        data["patient_id"] = patient_ids
    data["recommended_action_id"] = results["recommended_action"]
    data["recommended_action"] = [anames.get(int(a), f"action_{a}") for a in results["recommended_action"]]
    data["confidence_score"] = results["confidence_score"]
    data["state_value"] = results["v_values"].squeeze()

    for a in range(n_actions):
        name = anames.get(a, f"action_{a}")
        data[f"pi_{name}"] = results["pi"][:, a]
        data[f"conf_{name}"] = results["confidence"][:, a]

    df = pl.DataFrame(data)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        df.write_parquet(str(out))
    else:
        df.write_csv(str(out))
    print(f"  Exported {n:,} rows to {out}")
