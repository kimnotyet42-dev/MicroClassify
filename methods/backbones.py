"""Common backbone interface returning global image embeddings of shape (N, D).

The downstream nearest-centroid, Graph-z, and whitening implementations share
this representation. DINOv2 and DINOv3 use the CLS token from Transformers;
PE-Core and the CLIP-family models use their global image embeddings.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from PIL import Image


DEFAULT_FEATURE_BATCH = {
    "dinov3": 8,
    "dinov2": 64,
    "pecore": 64,
    "openai_clip": 32,
    "bioclip": 32,
}


def recommended_batch_size(name: str) -> int:
    if name not in DEFAULT_FEATURE_BATCH:
        raise KeyError("No reference batch size for {0}".format(name))
    return DEFAULT_FEATURE_BATCH[name]


class Backbone:
    cache_signature = "base-v1"

    def __init__(
        self,
        name: str,
        model_path: Path,
        image_size: int,
        device: str,
        pe_repo: Optional[Path] = None,
    ):
        self.name = name
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = device
        self.pe_repo = pe_repo
        self.model = None

    def load(self):
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, images: List[Image.Image]) -> np.ndarray:
        """Return unnormalized float32 global embeddings of shape (B, D)."""
        raise NotImplementedError


class HFViTBackbone(Backbone):
    cache_signature = "hf-official-processor-cls-v1"
    """Transformers AutoModel with the official processor and CLS token."""
    def load(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(str(self.model_path))
        for attr in ["do_center_crop"]:
            if hasattr(self.processor, attr):
                setattr(self.processor, attr, False)
        self.model = AutoModel.from_pretrained(str(self.model_path)).to(self.device).eval()

    @torch.no_grad()
    def embed(self, images):
        imgs = [image.convert("RGB") for image in images]
        inp = self.processor(
            images=imgs,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )
        inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        out = self.model(**inp)
        return out.last_hidden_state[:, 0, :].float().cpu().numpy()   # CLS token


class PECoreBackbone(Backbone):
    cache_signature = "pecore-official-preprocess-v1"
    """Meta Perception Encoder global image embedding.

    With ``pe_repo`` set, load a Meta-format PE-Core-L14-336.pt checkpoint
    through the official perception_models implementation.
    Otherwise, load an OpenCLIP-remapped checkpoint through open_clip.
    """
    def load(self):
        official_repo = str(self.pe_repo) if self.pe_repo is not None else ""
        if official_repo:
            if official_repo not in sys.path:
                sys.path.insert(0, official_repo)
            from core.vision_encoder import pe
            from core.vision_encoder import transforms as pe_transforms

            self.model = pe.VisionTransformer.from_config(
                "PE-Core-L14-336",
                pretrained=True,
                checkpoint_path=str(self.model_path),
            )
            self.preprocess = pe_transforms.get_image_transform(self.model.image_size)
            self.model = self.model.to(self.device).eval()
            self._official_meta = True
            return

        import open_clip
        # OpenCLIP includes the PE-Core-L-14-336 architecture; the configured
        # checkpoint must use OpenCLIP's remapped format on this path.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "PE-Core-L-14-336", pretrained=str(self.model_path))
        self.model = self.model.to(self.device).eval()
        self._official_meta = False

    @torch.no_grad()
    def embed(self, images):
        px = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        if self._official_meta:
            if self.device.startswith("cuda"):
                with torch.autocast("cuda"):
                    feat = self.model(px)
            else:
                feat = self.model(px)
        else:
            feat = self.model.encode_image(px)
        return feat.float().cpu().numpy()


class OpenClipImageBackbone(Backbone):
    """OpenAI CLIP or BioCLIP image encoder with the official preprocessing."""

    cache_signature = "openclip-autocast-gpu-l2norm-v1"

    def load(self):
        import open_clip

        if self.name == "openai_clip":
            self.model = open_clip.load_openai_model(
                str(self.model_path),
                precision="fp32",
                device=self.device,
            )
            self.preprocess = open_clip.image_transform(
                self.model.visual.image_size,
                is_train=False,
                mean=self.model.visual.image_mean,
                std=self.model.visual.image_std,
            )
        elif self.name == "bioclip":
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained=str(self.model_path),
                device=self.device,
            )
        else:
            raise ValueError("Unsupported CLIP-family backbone: {0}".format(self.name))
        self.model.eval()

    @torch.no_grad()
    def embed(self, images):
        import torch.nn.functional as functional

        pixels = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images]
        ).to(self.device)
        device_type = str(self.device).split(":", 1)[0]
        with torch.autocast(
            device_type=device_type,
            enabled=device_type == "cuda",
        ):
            features = functional.normalize(self.model.encode_image(pixels), dim=-1)
        return features.float().cpu().numpy()


def build_backbone(
    name: str,
    device: str,
    model_path: Optional[Path] = None,
    image_size: Optional[int] = None,
    pe_repo: Optional[Path] = None,
) -> Backbone:
    """Build a frozen backbone using an explicitly configured checkpoint path."""
    # The cross-backbone comparison uses 224 pixels for DINO models. The
    # separate 512-pixel DINOv3 result is evaluated by its dedicated protocol.
    cfgs = {
        "dinov3": (HFViTBackbone, 224),
        "dinov2": (HFViTBackbone, 224),
        # PE-Core's official preprocessing fixes its native 336-pixel input.
        "pecore": (PECoreBackbone, 336),
        "openai_clip": (OpenClipImageBackbone, 224),
        "bioclip": (OpenClipImageBackbone, 224),
    }
    if name not in cfgs:
        raise ValueError("Unknown backbone {0}; choose from {1}".format(name, list(cfgs)))
    cls, default_sz = cfgs[name]
    if model_path is None:
        raise RuntimeError("No checkpoint path configured for {0}".format(name))
    path = Path(model_path)
    return cls(name, path, image_size or default_sz, device, pe_repo=pe_repo)
