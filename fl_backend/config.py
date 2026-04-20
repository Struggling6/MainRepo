import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
from transformers import PatchTSTConfig as HF_PatchTSTConfig # We have to extend the HuggingFace config
from models import SupervisedTransformerCNN, LSTMModel, MLPModel
# ── Task configs ─────────────────────────────────────────────────────── #

@dataclass
class BinaryClassificationConfig:
    name: str = "binary_classification"

@dataclass
class AnomalyDetectionConfig:
    name: str = "anomaly_detection"

@dataclass
class MultiClassClassificationConfig:
    name: str = "multi_class_classification"


# ── Model configs ─────────────────────────────────────────────────────── #

@dataclass
class CNNTransformerConfig:
    name:           str   = "supervised_cnn_transformer"
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        type  = nn.BCEWithLogitsLoss

    def build(self, input_dim: int) -> nn.Module:
        return SupervisedTransformerCNN(
            in_channels=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )


@dataclass
class LSTMConfig:
    name:            str       = "lstm"
    hidden_size:     int       = 128
    num_layers:      int       = 2
    dropout:         float     = 0.3
    num_classes:     int       = 1
    pos_weight_cap:  float     = 10.0
    loss_fn:         type      = nn.BCEWithLogitsLoss

    def build(self, input_dim: int) -> nn.Module:
        return LSTMModel(
            in_channels=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

@dataclass
class MLPConfig:
    name:           str   = "mlp"
    hidden_size:    int   = 128
    num_layers:     int   = 2
    dropout:        float = 0.3
    num_classes:    int   = 1
    pos_weight_cap: float = 10.0
    loss_fn:        type  = nn.BCEWithLogitsLoss

    def build(self, input_dim: int) -> nn.Module:
        return MLPModel(
            in_channels=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

@dataclass
class PatchTSTConfig(HF_PatchTSTConfig):
    name:                   str = "patchtst"
    num_input_channels:     int         = 1
    context_length:         int         = 32       
    patch_length:           int         = 16
    patch_stride:           int         = 1
    d_model:                int         = 128
    num_attention_heads:    int         = 16
    num_hidden_layers:      int         = 3
    ffn_dim:                int         = 256
    dropout:                float       = 0.2
    head_dropout:           float       = 0.2
    channel_attention:      bool        = True
    loss:                   str         = "mse"
    attention_dropout:      float       = 0.0
    positional_dropout:     float       = 0.0
    pre_norm:               bool        = True
    norm_type:              Literal["batchnorm", "layernorm"] | None = "batchnorm"

    # Our own parameters (not in HuggingFace config) needed for our training loopS
    nhead:                  int         = num_attention_heads
    num_layers:             int         = num_hidden_layers

# ── Data configs ──────────────────────────────────────────────────────── #

@dataclass
class LeadCSVConfig:
    name:          str  = "lead_csv"
    file_path:     Path = Path("datasets/LEAD/train_features.csv") #used for shared mode, ignored for local mode, should be the large dataset csv
    data_dir            = Path("datasets/LEAD")
    file_pattern        = "data{client_index}.csv"
    target:        str  = "anomaly"
    batch_size:    int  = 64
    num_classes:   int  = 2
    task_name:     str  = "binary_classification"
    test_split:   float = 0.2
    seed:          int  = 42

@dataclass
class PowerGridCSVConfig:
    name:         str   = "powergrid_csv"
    file_path:    Path  = Path("datasets/EPIC/Scenario_1/EpicLog_noisy.csv")
    clean_path:   Path  = Path("datasets/EPIC/Scenario_1/EpicLog_clean.csv")
    target:       str   = "marker"
    batch_size:   int   = 32
    test_split:   float = 0.2
    normalize:    bool  = True
    noise_level:  float = 0.7
    seed:         int   = 42

# ── Training config ───────────────────────────────────────────────────── #

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    local_epochs:  int   = 2
    patience:      int   = 10


# ── Federation config ─────────────────────────────────────────────────── #

@dataclass
class FederationConfig:
    partition_mode:    str   = "shared" # local or shared
    num_rounds:        int   = 2
    num_clients:       int   = 1
    fraction_fit:      float = 1.0
    fraction_evaluate: float = 1.0
    proximal_mu:       float = 0.5

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
    model:      CNNTransformerConfig       = field(default_factory=CNNTransformerConfig)
    data:       LeadCSVConfig              = field(default_factory=LeadCSVConfig)
    training:   TrainingConfig             = field(default_factory=TrainingConfig)
    federation: FederationConfig           = field(default_factory=FederationConfig)
    evaluation: EvaluationConfig           = field(default_factory=EvaluationConfig)

# ── Active experiment ─────────────────────────────────────────────────── #
# This is the single line you change when switching experiments

# Example: switch to PowerGrid dataset with LSTM
# CONFIG = ExperimentConfig(
#     task=AnomalyDetectionConfig(),
#     model=LSTMConfig(hidden_size=256),
#     data=PowerGridCSVConfig(num_clients=3),
#     training=TrainingConfig(learning_rate=5e-4),
#     federation=FederationConfig(num_rounds=10),
#     evaluation=EvaluationConfig(threshold=0.3),
# )

CONFIG = ExperimentConfig(
    task=BinaryClassificationConfig(),
    model=PatchTSTConfig(),
    data=LeadCSVConfig(num_clients=4),
    training=TrainingConfig(learning_rate=1e-4, local_epochs=20),
    federation=FederationConfig(num_rounds=5),
    evaluation=EvaluationConfig(threshold=0.5),
)

