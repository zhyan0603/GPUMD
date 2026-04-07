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
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

MACE_MAGIC = 0x4D414345
MACE_VERSION = 1


def _load_checkpoint_any(path: Path) -> Any:
    load_errors = []

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch without weights_only support.
        pass
    except Exception as exc:
        load_errors.append(f"torch.load(weights_only=True) failed: {exc}")

    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # Older torch version signature.
        try:
            return torch.load(str(path), map_location="cpu")
        except Exception as exc:
            load_errors.append(f"torch.load(default) failed: {exc}")
    except Exception as exc:
        load_errors.append(f"torch.load(weights_only=False) failed: {exc}")

    try:
        return torch.jit.load(str(path), map_location="cpu")
    except Exception as exc:
        load_errors.append(f"torch.jit.load failed: {exc}")

    raise RuntimeError(
        "Failed to load checkpoint as state_dict, full model, or TorchScript.\n"
        + "\n".join(load_errors)
    )


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


def _find_by_token_sets(
    sd: Dict[str, torch.Tensor], token_sets: Iterable[Tuple[str, ...]], min_dim: int = 1
) -> Optional[Tuple[str, torch.Tensor]]:
    for token_set in token_sets:
        toks = tuple(t.lower() for t in token_set)
        for k, v in sd.items():
            kl = k.lower()
            if isinstance(v, torch.Tensor) and v.dim() >= min_dim and all(t in kl for t in toks):
                return k, v
    return None


def _format_tensor_summary(sd: Dict[str, torch.Tensor], limit: int = 80) -> str:
    lines = []
    for i, (k, v) in enumerate(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        lines.append(f"  - {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        if i + 1 >= limit:
            break
    return "\n".join(lines) if lines else "  (no tensor entries found)"


def _extract(sd: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> Dict[str, Any]:
    # species embedding
    emb_candidates: List[Tuple[str, torch.Tensor]] = []
    emb_match = _find_by_token_sets(
        sd,
        (
            ("node_embedding", "weight"),
            ("species_embedding",),
            ("atomic_embedding",),
            ("embedding", "weight"),
        ),
        min_dim=2,
    )
    if emb_match is not None:
        emb_candidates.append(emb_match)

    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2:
            kl = k.lower()
            if "embed" in kl or "species" in kl or "node" in kl:
                emb_candidates.append((k, v))
    if not emb_candidates:
        # fallback: smallest 2D tensor
        all_2d = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.dim() == 2]
        if all_2d:
            emb_candidates = all_2d
        else:
            raise RuntimeError("Failed to find a 2D species embedding tensor candidate.")
    emb_key, species_embedding = sorted(emb_candidates, key=lambda x: x[1].numel())[0]
    species_embedding = species_embedding.detach().cpu().float().contiguous()
    num_species, num_channels = species_embedding.shape

    # radial weights: choose first radial-like weight and reshape to [I,C,R]
    radial_match = _find_by_token_sets(
        sd,
        (
            ("radial_embedding", "weight"),
            ("radial", "weight"),
            ("bessel", "weight"),
            ("edge", "radial", "weight"),
            ("interaction", "radial", "weight"),
        ),
        min_dim=2,
    )
    if radial_match is None:
        all_3d = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.dim() == 3]
        if all_3d:
            radial_match = sorted(all_3d, key=lambda x: x[1].numel(), reverse=True)[0]
    if radial_match is None:
        # fallback to interaction-like 2D weight
        all_2d_interactions = []
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and "weight" in k.lower():
                kl = k.lower()
                if "interactions" in kl or "message" in kl or "radial" in kl:
                    all_2d_interactions.append((k, v))
        if all_2d_interactions:
            radial_match = sorted(all_2d_interactions, key=lambda x: x[1].numel(), reverse=True)[0]
    if radial_match is None:
        raise RuntimeError("Failed to find radial weight tensor candidate.")

    radial_key, radial_w = radial_match
    radial_w = radial_w.detach().cpu().float().contiguous()
    if radial_w.dim() == 1:
        radial_w = radial_w.view(1, 1, -1)
    elif radial_w.dim() == 2:
        # [C, R] -> [1, C, R]
        if radial_w.shape[0] == num_channels:
            radial_w = radial_w.unsqueeze(0)
        elif radial_w.shape[1] == num_channels:
            radial_w = radial_w.transpose(0, 1).unsqueeze(0).contiguous()
        else:
            radial_w = radial_w.unsqueeze(0)
    elif radial_w.dim() > 3:
        radial_w = radial_w.view(radial_w.shape[0], radial_w.shape[1], -1)
    if radial_w.shape[1] != num_channels and radial_w.shape[2] == num_channels:
        radial_w = radial_w.transpose(1, 2).contiguous()
    if radial_w.shape[1] != num_channels:
        # best-effort transpose fallback
        c_old = radial_w.shape[1]
        if c_old > num_channels:
            radial_w = radial_w[:, :num_channels, :].contiguous()
        else:
            repeat = (num_channels + c_old - 1) // c_old
            radial_w = radial_w.repeat(1, repeat, 1)[:, :num_channels, :].contiguous()
    num_interactions, _, num_radial = radial_w.shape

    # readout
    ro_match = _find_by_token_sets(
        sd,
        (
            ("readout", "linear", "weight"),
            ("readout", "weight"),
            ("output", "weight"),
            ("final", "weight"),
        ),
        min_dim=1,
    )
    if ro_match is None:
        # fallback: 2D weights with one output unit
        candidates = []
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and "weight" in k.lower():
                if 1 in v.shape:
                    candidates.append((k, v))
        if candidates:
            ro_match = sorted(candidates, key=lambda x: x[1].numel())[0]
    if ro_match is None:
        raise RuntimeError("Failed to find readout weight tensor candidate.")
    ro_key, ro = ro_match
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
        "species_embedding": species_embedding.reshape(-1).numpy(),
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

    try:
        obj = _load_checkpoint_any(args.input_model)
        state_dict, meta, flags = _as_state_dict(obj)
        payload = _extract(state_dict, meta)
    except Exception as exc:
        print(f"[ERROR] Conversion failed: {exc}", file=sys.stderr)
        try:
            obj_debug = _load_checkpoint_any(args.input_model)
            sd_debug, _, _ = _as_state_dict(obj_debug)
            print("[DEBUG] First tensor keys/shapes:", file=sys.stderr)
            print(_format_tensor_summary(sd_debug), file=sys.stderr)
        except Exception as exc2:
            print(f"[DEBUG] Failed to summarize checkpoint tensors: {exc2}", file=sys.stderr)
        sys.exit(1)

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
