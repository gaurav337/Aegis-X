"""Configuration module housing all application configuration state.

This module defines standard typed configuration classes that serve
as the single source of truth for configuration across Aegis-X.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional  

from utils.thresholds import (
    WEIGHT_SIGLIP, WEIGHT_SBI, WEIGHT_FREQNET, WEIGHT_RPPG,
    WEIGHT_DCT, WEIGHT_GEOMETRY, WEIGHT_ILLUMINATION,
    REAL_THRESHOLD, FAKE_THRESHOLD, EARLY_STOP_CONFIDENCE,
    SBI_SKIP_CLIP_THRESHOLD
)

# Load environment variables early
load_dotenv()

@dataclass
class ModelPaths:
    """Paths to models and weights used by the system."""
    # Constructed using pathlib but cast to str to match specifications
    phi3_model: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "phi3")
    clip_adapter_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "clip_adapter.pt")
    sbi_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "sbi.pt")
    freqnet_weights: str = str(Path(os.getenv("AEGIS_MODEL_DIR", "models/")) / "freqnet.pt")

@dataclass
class AgentConfig:
    """Configuration for the local LLM agent and orchestration."""
    max_retries: int = int(os.getenv("AGENT_MAX_RETRIES", "2"))
    llm_timeout: int = int(os.getenv("AGENT_LLM_TIMEOUT", "120"))
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    
    # === OLLAMA SPECIFIC ===
    ollama_model_name: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    # C1: Omit for Ollama default (5min) - best for batch scanning
    ollama_keep_alive: Optional[int] = None
    
    # === GENERATION SETTINGS ===
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_seed: int = int(os.getenv("LLM_SEED", "42"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    llm_context_window: int = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))

@dataclass
class EnsembleWeights:
    """Weights for the final ensemble verdict calculation."""
    clip_adapter: float = WEIGHT_SIGLIP
    sbi: float = WEIGHT_SBI
    freqnet: float = WEIGHT_FREQNET
    rppg: float = WEIGHT_RPPG
    dct: float = WEIGHT_DCT
    geometry: float = WEIGHT_GEOMETRY
    illumination: float = WEIGHT_ILLUMINATION

@dataclass
class ThresholdConfig:
    """Thresholds for verdicts and tool routing."""
    real_threshold: float = REAL_THRESHOLD
    fake_threshold: float = FAKE_THRESHOLD
    early_stop_confidence: float = EARLY_STOP_CONFIDENCE
    sbi_skip_clip_threshold: float = SBI_SKIP_CLIP_THRESHOLD

@dataclass
class PreprocessingConfig:
    """Configuration for media preprocessing and patching."""
    face_crop_size: int = 224
    sbi_crop_size: int = 380
    native_patch_size: int = 224
    max_video_frames: int = 300
    min_video_frames: int = 90
    extract_fps: int = 30
    video_backend: str = "auto"
    quality_snipe_enabled: bool = True
    quality_snipe_samples: int = 5

@dataclass
class AegisConfig:
    """Master configuration class grouping all subsystem configs."""
    models: ModelPaths = field(default_factory=ModelPaths)
    agent: AgentConfig = field(default_factory=AgentConfig)
    weights: EnsembleWeights = field(default_factory=EnsembleWeights)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

