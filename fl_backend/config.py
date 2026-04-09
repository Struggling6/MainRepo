from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ── Task configs ─────────────────────────────────────────────────────── #

@dataclass
class BinaryClassificationConfig:
    name: str = "binary_classification"

@dataclass
class AnomalyDetectionConfig:
    name: str = "anomaly_detection"


# ── Model configs ─────────────────────────────────────────────────────── #

@dataclass
class CNNTransformerConfig:
    name:           str   = "supervised_cnn_transformer"
    d_model:        int   = 128
    num_heads:      int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    in_channels:    int   = 1
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1  # binary classification

@dataclass
class LSTMConfig:
    name:        str   = "lstm"
    hidden_size: int   = 128
    num_layers:  int   = 2
    dropout:     float = 0.3
    in_channels: int   = 1


# ── Data configs ──────────────────────────────────────────────────────── #

@dataclass
class LeadCSVConfig:
    name:         str  = "lead_csv"
    file_path:    Path = Path("datasets/LEAD/train_features.csv")
    target:       str  = "anomaly"
    batch_size:   int  = 64
    num_clients:  int  = 1
    test_split:   float = 0.2
    seed:         int  = 42

@dataclass
class PowerGridCSVConfig:
    name:         str   = "powergrid_csv"
    file_path:    Path  = Path("datasets/EPIC/Scenario_1/EpicLog_noisy.csv")
    clean_path:   Path  = Path("datasets/EPIC/Scenario_1/EpicLog_clean.csv")
    target:       str   = "marker"
    batch_size:   int   = 32
    num_clients:  int   = 1
    test_split:   float = 0.2
    normalize:    bool  = True
    noise_level:  float = 0.7
    seed:         int   = 42


# ── Training config ───────────────────────────────────────────────────── #

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    local_epochs:  int   = 30
    patience:      int   = 10


# ── Federation config ─────────────────────────────────────────────────── #

@dataclass
class FederationConfig:
    num_rounds:        int   = 5
    fraction_fit:      float = 1.0
    fraction_evaluate: float = 1.0


# ── Top-level experiment config ───────────────────────────────────────── #

@dataclass
class ExperimentConfig:
    task:       BinaryClassificationConfig = field(default_factory=BinaryClassificationConfig)
    model:      CNNTransformerConfig       = field(default_factory=CNNTransformerConfig)
    data:       LeadCSVConfig              = field(default_factory=LeadCSVConfig)
    training:   TrainingConfig             = field(default_factory=TrainingConfig)
    federation: FederationConfig           = field(default_factory=FederationConfig)


# ── Active experiment ─────────────────────────────────────────────────── #
# This is the single line you change when switching experiments

CONFIG = ExperimentConfig()

# Example: switch to PowerGrid dataset with LSTM
# CONFIG = ExperimentConfig(
#     task=AnomalyDetectionConfig(),
#     model=LSTMConfig(hidden_size=256),
#     data=PowerGridCSVConfig(num_clients=3),
#     training=TrainingConfig(learning_rate=5e-4),
# )