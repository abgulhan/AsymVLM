"""Model-family adapters for the shared scorer trainer."""

from .base import PreparedBatch, PrunedBatch, VLMAdapter
from .llava import LlavaAdapter
from .llava_next import LlavaNextAdapter
from .phi3_vision import Phi3VisionAdapter
from .qwen2_vl import Qwen2VLAdapter
from .qwen2_5_vl import Qwen2_5_VLAdapter

__all__ = [
    "LlavaAdapter",
    "LlavaNextAdapter",
    "Phi3VisionAdapter",
    "Qwen2VLAdapter",
    "Qwen2_5_VLAdapter",
    "PreparedBatch",
    "PrunedBatch",
    "VLMAdapter",
]
