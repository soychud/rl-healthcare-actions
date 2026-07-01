import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List
from src.config import LAB_FEATURES, VITAL_FEATURES, auto_feature_cols


def _all_feature_cols() -> list:
    return list(LAB_FEATURES.keys()) + list(VITAL_FEATURES.keys())


def _state_columns(df: pl.DataFrame) -> List[str]:
    """Auto-detect state columns from whatever features are present."""
    feat_cols = _all_feature_cols()
    has_config = any(f in df.columns for f in feat_cols)
    if not has_config:
        feat_cols = auto_feature_cols(df)
    z_cols = [f"{c}_z" for c in feat_cols if f"{c}_z" in df.columns]
    if not z_cols:
        z_cols = []
    all_missing = [f"{c}_missing" for c in feat_cols if f"{c}_missing" in df.columns]
    demo = []
    for c in ["anchor_age", "gender_male", "time_sin", "time_cos"]:
        if c in df.columns:
            demo.append(c)
    delta_cols = sorted(c for c in df.columns if c.startswith("delta_"))
    return z_cols + all_missing + demo + delta_cols


class TrajectoryDataset(Dataset):
    def __init__(self, parquet_path: str, max_len: Optional[int] = None):
        df = pl.read_parquet(parquet_path)
        self.state_cols = _state_columns(df)
        self.state_dim = len(self.state_cols)
        self.df = df
        adms = df.group_by("hadm_id").agg(pl.len().alias("tlen"))
        if max_len is not None:
            adms = adms.filter(pl.col("tlen") <= max_len)
        self.hadm_ids = adms.sort("hadm_id")["hadm_id"].to_list()
        self._cache = {}

    def __len__(self):
        return len(self.hadm_ids)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        hadm = self.hadm_ids[idx]
        adm = self.df.filter(pl.col("hadm_id") == hadm).sort("bin_idx")
        states = adm.select(self.state_cols).fill_null(0.0).to_numpy().astype(np.float32)
        actions = adm["action_id"].to_numpy().astype(np.int64)
        rewards = adm["reward"].to_numpy().astype(np.float32)
        result = {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rewards": torch.from_numpy(rewards),
            "hadm_id": hadm,
            "length": len(actions),
        }
        if len(self._cache) < 5000:
            self._cache[idx] = result
        return result


class FlatDataset(Dataset):
    def __init__(self, parquet_path: str, gamma: float = 0.99):
        df = pl.read_parquet(parquet_path)
        self.state_cols = _state_columns(df)
        self.state_dim = len(self.state_cols)
        df = df.sort("hadm_id", "bin_idx")
        states = df.select(self.state_cols).fill_null(0.0).to_numpy().astype(np.float32)
        actions = df["action_id"].to_numpy().astype(np.int64)
        rewards = df["reward"].to_numpy().astype(np.float32)

        next_states = np.zeros_like(states)
        next_states[:-1] = states[1:]
        dones = np.zeros(len(df), dtype=np.float32)
        hadm = df["hadm_id"].to_numpy()
        hadm_next = np.roll(hadm, -1)
        dones[:-1] = (hadm[:-1] != hadm_next[:-1]).astype(np.float32)
        dones[-1] = 1.0
        next_states[dones.astype(bool)] = 0.0

        discounts = np.zeros(len(df), dtype=np.float32)
        adms, counts = np.unique(hadm, return_counts=True)
        offset = 0
        for c in counts:
            for t in range(c):
                discounts[offset + t] = gamma ** t
            offset += c

        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)
        self.rewards = torch.from_numpy(rewards)
        self.next_states = torch.from_numpy(next_states)
        self.dones = torch.from_numpy(dones)
        self.discounts = torch.from_numpy(discounts)

    def to_device(self, device: torch.device):
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)
        self.discounts = self.discounts.to(device)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_state": self.next_states[idx],
            "done": self.dones[idx],
            "discount": self.discounts[idx],
        }
