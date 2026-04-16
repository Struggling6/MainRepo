from dataclasses import dataclass, field
from pathlib import Path
import torch.nn as nn


# ── Task configs ─────────────────────────────────────────────────────── #

@dataclass
class BinaryClassificationConfig:
    name: str = "binary_classification"


# ── Model configs ─────────────────────────────────────────────────────── #

@dataclass
class CNNTransformerConfig:
    name:           str   = "supervised_cnn_transformer"
    d_model:        int   = 128
    nhead:          int   = 4 # number of attention heads (nhead is PyTorch's parameter name)
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1  # binary classification
    loss_fn:        type  = nn.BCEWithLogitsLoss  # default loss function for binary classification

@dataclass
class LSTMConfig:
    name:        str   = "lstm"
    hidden_size: int   = 128
    num_layers:  int   = 2
    dropout:     float = 0.3
    loss_fn:     type  = nn.BCEWithLogitsLoss


@dataclass
class TransformerConfig:
    name:           str   = "transformer"
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0 
    num_classes:    int   = 1
    loss_fn:        type  = nn.BCEWithLogitsLoss

# ── Data configs ──────────────────────────────────────────────────────── #

@dataclass
class LeadCSVConfig:
    name:          str  = "lead_csv"
    file_path:     Path = Path("datasets/LEAD/train_features.csv")
    target:        str  = "anomaly"
    batch_size:    int  = 64
    test_split:   float = 0.4
    num_clients:   int  = 1
    seed:          int  = 42
    partition_mode: str = "shared"

@dataclass
class PowerGridCSVConfig:
    name:         str   = "powergrid_csv"
    file_path:    Path  = Path("datasets/EPIC/Scenario_1/EpicLog_noisy.csv")
    clean_path:   Path  = Path("datasets/EPIC/Scenario_1/EpicLog_clean.csv")
    target:       str   = "marker"
    batch_size:   int   = 32
    num_clients:  int   = 2
    test_split:   float = 0.2
    normalize:    bool  = True
    noise_level:  float = 0.7
    seed:         int   = 42
    partition_mode: str = "shared"


# ── Training config ───────────────────────────────────────────────────── #

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    local_epochs:  int   = 5
    patience:      int   = 10


# ── Federation config ─────────────────────────────────────────────────── #

@dataclass
class FederationConfig:
    num_rounds:        int   = 5
    fraction_fit:      float = 1.0
    fraction_evaluate: float = 1.0

# ── Evaluation config ─────────────────────────────────────────────────── #

@dataclass
class EvaluationConfig:
    model_path:   Path  = Path("checkpoints/model.pt")
    test_path:    Path  = Path("datasets/LEAD/test_features.csv")
    target:       str   = "anomaly"
    batch_size:   int   = 64
    threshold:    float = 0.5   # decision threshold — override with best_thresh from training
    input_dim:    int   = 0     # set after data loading

# ── Top-level experiment config ───────────────────────────────────────── #

@dataclass
class ExperimentConfig:
    task:       BinaryClassificationConfig = field(default_factory=BinaryClassificationConfig)
    model:      TransformerConfig       = field(default_factory=TransformerConfig)
    data:       LeadCSVConfig             = field(default_factory=LeadCSVConfig)
    training:   TrainingConfig             = field(default_factory=TrainingConfig)
    federation: FederationConfig           = field(default_factory=FederationConfig)
    evaluation: EvaluationConfig           = field(default_factory=EvaluationConfig)

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
