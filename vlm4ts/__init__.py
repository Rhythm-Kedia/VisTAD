"""VLM4TS Engine: Two-Stage Vision-Language Modeling for Time Series Anomaly Detection.

Implements visual coarse-screening (ViT4TS) and multimodal verification (VLM4TS)
tailored for cyber-physical and complex non-spike time-series anomalies.
"""

from .renderer import render_vlm4ts_image, detect_channel_types
from .vit_screener import ViT4TSScreener
from .vlm_reasoner import VLM4TSReasoner

__all__ = ["render_vlm4ts_image", "detect_channel_types", "ViT4TSScreener", "VLM4TSReasoner"]
