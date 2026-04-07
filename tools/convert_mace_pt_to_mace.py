#!/usr/bin/env python3
"""
One-time MACE PyTorch checkpoint -> native .mace converter.

Runtime dependency policy:
  - torch is required only in this offline converter.
  - gpumd_mace runtime has zero Python / zero torch dependency.

Binary .mace format (little-endian):
  Header (fixed 76 bytes):
    uint32 magic            = 0x4D414345 ('MACE')
    uint32 version          = 1
    uint32 flags            (bit0: ScaleShiftMACE present, bit1: TorchScript input)
    uint32 dtype            = 0 (float32 payload)
    uint32 num_species
    uint32 num_channels
    uint32 num_radial
    uint32 num_interactions
    uint32 l_max
    uint32 max_neighbors
    float32 r_max
    float32 cutoff_p
    float32 cutoff_q
    float32 scale
    float32 shift
    uint32 reserved0
    uint32 reserved1
    uint32 reserved2
    uint32 reserved3

  Tensor sections:
    [uint64 n0][float32[n0]]  species_embedding (num_species * num_channels)
    [uint64 n1][float32[n1]]  radial_weights    (num_interactions * num_channels * num_radial)
    [uint64 n2][float32[n2]]  readout_weight    (num_channels)
    [uint64 n3][float32[n3]]  readout_bias      (1)

This converter targets invariant-only inference weights for native CUDA path.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

MACE_MAGIC = 0x4D414345
MACE_VERSION = 1


def _as_state_dict(obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], int]:
    flags = 0
    meta: Dict[str, Any] = {}

    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            sd = obj.state_dict()
            flags |= 0x2
            if hasattr(obj, "r_max"):
                meta["r_max"] = float(obj.r_max)
            if hasattr(obj, "max_ell"):
                meta["l_max"] = int(obj.max_ell)
            return sd, meta, flags
        except Exception:
            pass

    if isinstance(obj, dict):
        for k in ("model", "ema_model", "module"):
            if k in obj and hasattr(obj[k], "state_dict"):
                sd = obj[k].state_dict()
                nested = obj.get("config", {})
                if isinstance(nested, dict):
                    meta.update(nested)
                return sd, meta, flags
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if "config" in obj and isinstance(obj["config"], dict):
                meta.update(obj["config"])
            return sd, meta, flags
        if all(isinstance(k, str) for k in obj.keys()) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return obj, meta, flags

    raise RuntimeError("Unsupported input: could not obtain a state_dict.")


def _find_first(
    sd: Dict[str, torch.Tensor],
    required_tokens: Iterable[str],
    min_dim: int = 1,
) -> Tuple[str, torch.Tensor]:
    toks = tuple(t.lower() for t in required_tokens)
    for k, v in sd.items():
        kl = k.lower()
        if all(t in kl for t in toks) and isinstance(v, torch.Tensor) and v.dim() >= min_dim:
            return k, v
    raise RuntimeError(f"Could not find tensor with tokens {required_tokens}.")


def _extract(sd: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> Dict[str, Any]:
    # species embedding
    emb_candidates = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.dim() != 2:
            continue
        kl = k.lower()
        if "embed" in kl or "species" in kl or "node" in kl:
            emb_candidates.append((k, v))
    if not emb_candidates:
        raise RuntimeError("Failed to find species embedding tensor.")
    emb_key, emb = sorted(emb_candidates, key=lambda x: x[1].numel())[0]
    emb = emb.detach().cpu().float().contiguous()
    num_species, num_channels = emb.shape

    # radial weights: choose first radial-like weight and reshape to [I,C,R]
    radial_key, radial_w = _find_first(sd, ("radial", "weight"), min_dim=2)
    radial_w = radial_w.detach().cpu().float().contiguous()
    if radial_w.dim() == 2:
        # [C, R] -> [1, C, R]
        radial_w = radial_w.unsqueeze(0)
    elif radial_w.dim() > 3:
        radial_w = radial_w.view(radial_w.shape[0], radial_w.shape[1], -1)
    if radial_w.shape[1] != num_channels:
        # best-effort transpose fallback
        if radial_w.shape[2] == num_channels:
            radial_w = radial_w.transpose(1, 2).contiguous()
        else:
            radial_w = radial_w.reshape(radial_w.shape[0], num_channels, -1)
    num_interactions, _, num_radial = radial_w.shape

    # readout
    ro_key, ro = _find_first(sd, ("readout", "weight"), min_dim=1)
    ro = ro.detach().cpu().float().contiguous()
    if ro.dim() == 2:
        if ro.shape[0] == 1:
            ro = ro[0]
        elif ro.shape[1] == 1:
            ro = ro[:, 0]
        else:
            ro = ro.reshape(-1)
    if ro.numel() != num_channels:
        ro = ro.reshape(-1)[:num_channels].contiguous()

    # readout bias (optional in some exported models)
    ro_b = None
    for k, v in sd.items():
        kl = k.lower()
        if "readout" in kl and "bias" in kl and isinstance(v, torch.Tensor):
            ro_b = v.detach().cpu().float().reshape(-1)
            break
    if ro_b is None or ro_b.numel() == 0:
        ro_b = torch.zeros(1, dtype=torch.float32)
    else:
        ro_b = ro_b[:1].contiguous()

    r_max = float(meta.get("r_max", meta.get("cutoff", 5.0)))
    l_max = int(meta.get("max_ell", meta.get("l_max", 0)))
    max_neighbors = int(meta.get("max_num_neighbors", 256))
    cutoff_p = float(meta.get("cutoff_p", 6.0))
    cutoff_q = float(meta.get("cutoff_q", 6.0))
    scale = float(meta.get("scale", 1.0))
    shift = float(meta.get("shift", 0.0))

    return {
        "num_species": int(num_species),
        "num_channels": int(num_channels),
        "num_radial": int(num_radial),
        "num_interactions": int(num_interactions),
        "l_max": int(l_max),
        "max_neighbors": int(max_neighbors),
        "r_max": r_max,
        "cutoff_p": cutoff_p,
        "cutoff_q": cutoff_q,
        "scale": scale,
        "shift": shift,
        "species_embedding": emb.reshape(-1).numpy(),
        "radial_weights": radial_w.reshape(-1).numpy(),
        "readout_weight": ro.reshape(-1).numpy(),
        "readout_bias": ro_b.reshape(-1).numpy(),
        "debug_keys": (emb_key, radial_key, ro_key),
    }


def _write_tensor(f, arr) -> None:
    arr = arr.astype("float32", copy=False)
    f.write(struct.pack("<Q", int(arr.size)))
    if arr.size:
        f.write(arr.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MACE .pt/.model to native .mace format")
    parser.add_argument("input_model", type=Path, help="Path to MACE checkpoint (.pt/.model)")
    parser.add_argument("output_mace", type=Path, help="Path to output .mace file")
    args = parser.parse_args()

    obj = torch.load(str(args.input_model), map_location="cpu")
    state_dict, meta, flags = _as_state_dict(obj)
    payload = _extract(state_dict, meta)

    model_name = type(obj).__name__.lower()
    if "scaleshift" in model_name:
        flags |= 0x1

    header = struct.pack(
        "<10I5f4I",
        MACE_MAGIC,
        MACE_VERSION,
        flags,
        0,  # dtype float32
        payload["num_species"],
        payload["num_channels"],
        payload["num_radial"],
        payload["num_interactions"],
        payload["l_max"],
        payload["max_neighbors"],
        payload["r_max"],
        payload["cutoff_p"],
        payload["cutoff_q"],
        payload["scale"],
        payload["shift"],
        0,
        0,
        0,
        0,
    )

    args.output_mace.parent.mkdir(parents=True, exist_ok=True)
    with args.output_mace.open("wb") as f:
        f.write(header)
        _write_tensor(f, payload["species_embedding"])
        _write_tensor(f, payload["radial_weights"])
        _write_tensor(f, payload["readout_weight"])
        _write_tensor(f, payload["readout_bias"])

    emb_key, radial_key, ro_key = payload["debug_keys"]
    print("Wrote", args.output_mace)
    print("  species:", payload["num_species"])
    print("  channels:", payload["num_channels"])
    print("  radial:", payload["num_radial"])
    print("  interactions:", payload["num_interactions"])
    print("  selected tensors:")
    print("   - embedding:", emb_key)
    print("   - radial:", radial_key)
    print("   - readout:", ro_key)


if __name__ == "__main__":
    main()
