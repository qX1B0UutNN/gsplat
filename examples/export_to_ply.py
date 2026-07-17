# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a trained gsplat checkpoint to standard 3DGS PLY.
AI generated

Examples:
    python ext/gsplat/examples/export_to_ply.py \
        --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --output results/garden/point_cloud.ply
"""

import math
import os
import sys
from pathlib import Path
from typing import Mapping

import fire
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsplat import export_splats


REQUIRED_SPLAT_KEYS = ("means", "scales", "quats", "opacities")


class AppearanceOptModule(torch.nn.Module):
    """Minimal copy of simple_trainer's appearance module for color baking."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = [
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width),
            torch.nn.ReLU(inplace=True),
        ]
        for _ in range(mlp_depth - 1):
            layers.extend(
                [torch.nn.Linear(mlp_width, mlp_width), torch.nn.ReLU(inplace=True)]
            )
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor | None, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)
        embeds = embeds[:, None, :].expand(-1, N, -1)
        features = features[None, :, :].expand(C, -1, -1)
        dirs = F.normalize(dirs, dim=-1)
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        return self.color_head(h)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _torch_load(path: Path, map_location: torch.device) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _as_tensor_dict(value: object, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}.")
    tensor_dict = dict(value)
    non_tensors = [k for k, v in tensor_dict.items() if not torch.is_tensor(v)]
    if non_tensors:
        raise ValueError(f"{name} contains non-tensor entries: {non_tensors}")
    return tensor_dict


def _load_splats(
    payload: object,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if isinstance(payload, Mapping) and "splats" in payload:
        splats = _as_tensor_dict(payload["splats"], "checkpoint['splats']")
        app_state = payload.get("app_module")
        if app_state is not None:
            app_state = _as_tensor_dict(app_state, "checkpoint['app_module']")
        return splats, app_state

    splats = _as_tensor_dict(payload, "checkpoint")
    return splats, None


def _infer_sh_degree_from_state(
    splats: Mapping[str, torch.Tensor],
    app_state: Mapping[str, torch.Tensor],
) -> int:
    if "features" not in splats:
        raise ValueError("Appearance checkpoint is missing splats['features'].")
    if "embeds.weight" not in app_state or "color_head.0.weight" not in app_state:
        raise ValueError("Appearance checkpoint has an unsupported app_module state.")

    embed_dim = app_state["embeds.weight"].shape[1]
    feature_dim = splats["features"].shape[1]
    input_dim = app_state["color_head.0.weight"].shape[1]
    num_sh_bases = input_dim - embed_dim - feature_dim
    sh_degree = math.isqrt(num_sh_bases) - 1
    if (sh_degree + 1) ** 2 != num_sh_bases:
        raise ValueError(
            "Could not infer appearance SH degree from app_module input shape "
            f"{input_dim}, embed_dim {embed_dim}, feature_dim {feature_dim}."
        )
    return sh_degree


def _build_appearance_module(
    splats: Mapping[str, torch.Tensor],
    app_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[AppearanceOptModule, int]:
    sh_degree = _infer_sh_degree_from_state(splats, app_state)
    n_cameras, embed_dim = app_state["embeds.weight"].shape
    feature_dim = splats["features"].shape[1]
    mlp_width = app_state["color_head.0.weight"].shape[0]
    linear_weight_keys = [
        key
        for key in app_state
        if key.startswith("color_head.") and key.endswith(".weight")
    ]
    mlp_depth = len(linear_weight_keys) - 1
    if mlp_depth < 1:
        raise ValueError("Appearance checkpoint has no hidden MLP layers.")

    module = AppearanceOptModule(
        n=n_cameras,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        sh_degree=sh_degree,
        mlp_width=mlp_width,
        mlp_depth=mlp_depth,
    ).to(device)
    module.load_state_dict(app_state)
    module.eval()
    return module, sh_degree


def _select_sh(
    splats: Mapping[str, torch.Tensor],
    app_state: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if app_state is not None:
        if "colors" not in splats:
            raise ValueError("Appearance checkpoint is missing splats['colors'].")
        app_module, sh_degree = _build_appearance_module(splats, app_state, device)
        with torch.no_grad():
            features = splats["features"].to(device)
            colors = splats["colors"].to(device)
            dirs = torch.zeros_like(splats["means"][None, :, :], device=device)
            rgb = app_module(
                features=features,
                embed_ids=None,
                dirs=dirs,
                sh_degree=sh_degree,
            )
            rgb = torch.sigmoid(rgb + colors).squeeze(0).unsqueeze(1)
            sh0 = rgb_to_sh(rgb)
            shN = torch.empty((sh0.shape[0], 0, 3), device=device, dtype=sh0.dtype)
        return sh0, shN

    if "sh0" in splats and "shN" in splats:
        return splats["sh0"].to(device), splats["shN"].to(device)

    if "colors" in splats:
        colors = splats["colors"].to(device)
        if colors.ndim == 2 and colors.shape[-1] == 3:
            sh0 = colors.unsqueeze(1)
            shN = torch.empty(
                (colors.shape[0], 0, 3), device=device, dtype=colors.dtype
            )
            return sh0, shN
        if colors.ndim == 3 and colors.shape[-1] == 3:
            return colors[:, :1, :], colors[:, 1:, :]

    raise ValueError(
        "Checkpoint must contain sh0/shN, colors, or app_module-backed colors."
    )


def export_checkpoint_to_ply(
    ckpt_path: Path, output_path: Path, device: torch.device
) -> None:
    payload = _torch_load(ckpt_path, device)
    splats, app_state = _load_splats(payload)

    missing_keys = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
    if missing_keys:
        raise ValueError(f"Checkpoint is missing required splat keys: {missing_keys}")

    sh0, shN = _select_sh(splats, app_state, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=splats["means"].to(device),
        scales=splats["scales"].to(device),
        quats=splats["quats"].to(device),
        opacities=splats["opacities"].to(device),
        sh0=sh0,
        shN=shN,
        format="ply",
        save_to=str(output_path),
    )


def main(ckpt: str, output: str, device: str = "cpu") -> None:
    """Export a trained gsplat .pt checkpoint to standard 3DGS PLY."""
    ckpt_path = Path(ckpt)
    output_path = Path(output)
    export_checkpoint_to_ply(ckpt_path, output_path, torch.device(device))
    print(f"Exported {ckpt_path} to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
