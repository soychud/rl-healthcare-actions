"""REST API server for RL inference. Start with: python3 -m src.rl.server"""

import json
from typing import Optional, List
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.rl.inference import InferenceEnsemble, build_safety_mask
from src.config import N_ACTIONS, ACTION_BUNDLES

app = FastAPI(title="RL Healthcare Actions Inference", version="1.0.0")

_ensemble: Optional[InferenceEnsemble] = None
_action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()}


class PredictRequest(BaseModel):
    state: List[float] = Field(..., description="State vector")
    safety_labs: Optional[List[float]] = Field(None, description="Lab values for safety masking")
    lab_names: Optional[List[str]] = Field(None, description="Lab feature names matching SAFETY_CONSTRAINTS")
    hadm_id: Optional[int] = Field(None, description="Admission ID for diagnosis-based safety checks")


class PredictResponse(BaseModel):
    recommended_action: int
    recommended_action_name: str
    action_probabilities: List[float]
    action_confidence: List[float]
    q_values: List[float]
    q_std: List[float]
    state_value: float


class BatchPredictRequest(BaseModel):
    states: List[List[float]]
    safety_labs: Optional[List[List[float]]] = None
    lab_names: Optional[List[str]] = None
    hadm_ids: Optional[List[int]] = None


class BatchPredictResponse(BaseModel):
    recommendations: List[dict]


def get_ensemble() -> InferenceEnsemble:
    global _ensemble
    if _ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load first.")
    return _ensemble


@app.on_event("startup")
def _startup():
    pass


@app.get("/health")
def health():
    global _ensemble
    loaded = _ensemble is not None
    n_act = _ensemble.n_actions if _ensemble is not None else N_ACTIONS
    return {"status": "ok", "model_loaded": loaded, "n_actions": n_act}


@app.get("/actions")
def list_actions():
    global _action_names
    return {"actions": _action_names}


@app.post("/load")
def load_model(model_dir: str = "data/models", seeds: Optional[List[int]] = None, state_dim: int = 64):
    global _ensemble, _action_names
    _ensemble = InferenceEnsemble(state_dim=state_dim, model_dir=model_dir, seeds=seeds or [0, 1, 2, 3, 42])
    n_act = _ensemble.n_actions
    _action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()} if ACTION_BUNDLES else {i: f"action_{i}" for i in range(n_act)}
    return {"status": "loaded", "n_ensemble": _ensemble.n_ensemble, "state_dim": state_dim}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ens = get_ensemble()
    state = torch.tensor(req.state, dtype=torch.float32).unsqueeze(0)

    safety_mask = None
    if req.safety_labs is not None and req.lab_names is not None:
        labs_arr = np.array(req.safety_labs).reshape(1, -1)
        hadm_arr = np.array([req.hadm_id]) if req.hadm_id is not None else None
        safety_mask = build_safety_mask(
            n_rows=1, n_actions=ens.n_actions,
            labs=labs_arr, lab_names=req.lab_names,
            hadm_ids=hadm_arr,
        )

    result = ens.predict_safe(state, safety_mask=safety_mask)
    pi = result["pi"][0]
    return PredictResponse(
        recommended_action=int(pi.argmax()),
        recommended_action_name=_action_names.get(int(pi.argmax()), "unknown"),
        action_probabilities=pi.tolist(),
        action_confidence=result["confidence"][0].tolist(),
        q_values=result["q_values"][0].tolist(),
        q_std=result["q_std"][0].tolist(),
        state_value=float(result["v_values"][0][0]),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    ens = get_ensemble()
    states = torch.tensor(req.states, dtype=torch.float32)
    n = states.shape[0]

    safety_mask = None
    if req.safety_labs is not None and req.lab_names is not None:
        labs_arr = np.array(req.safety_labs).reshape(n, -1)
        hadm_arr = np.array(req.hadm_ids) if req.hadm_ids else None
        safety_mask = build_safety_mask(
            n_rows=n, n_actions=ens.n_actions,
            labs=labs_arr, lab_names=req.lab_names,
            hadm_ids=hadm_arr,
        )

    result = ens.predict_safe(states, safety_mask=safety_mask)
    recs = []
    for i in range(n):
        pi = result["pi"][i]
        recs.append({
            "recommended_action": int(pi.argmax()),
            "recommended_action_name": _action_names.get(int(pi.argmax()), "unknown"),
            "action_probabilities": pi.tolist(),
            "confidence": result["confidence"][i].tolist(),
            "q_values": result["q_values"][i].tolist(),
            "state_value": float(result["v_values"][i][0]),
        })
    return BatchPredictResponse(recommendations=recs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RL Healthcare Actions API server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 42])
    parser.add_argument("--preload", action="store_true", help="Load model on startup")
    args = parser.parse_args()

    if args.preload:
        _ensemble = InferenceEnsemble(state_dim=args.state_dim, model_dir=args.model_dir, seeds=args.seeds)
        _action_names = {k: v["name"] for k, v in ACTION_BUNDLES.items()} if ACTION_BUNDLES else {i: f"action_{i}" for i in range(_ensemble.n_actions)}
        print(f"Model loaded: {_ensemble.n_ensemble} ensemble members")

    uvicorn.run(app, host=args.host, port=args.port)
