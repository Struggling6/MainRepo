from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

def resolve_loss_fn(loss_fn):
    """Resolve a loss function name or class to a torch.nn loss class."""
    import torch.nn as nn

    if isinstance(loss_fn, str):
        resolved = getattr(nn, loss_fn, None)
        if resolved is None:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        return resolved

    return loss_fn

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
    batch_size:     int   = 64
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int):
        from models.supervised_cnn_transformer import SupervisedCNNTransformer

        return SupervisedCNNTransformer(
            in_channels=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )


@dataclass
class TransformerConfig:
    name:           str   = "transformer"
    batch_size:   int     = 64
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        str   = "BCEWithLogitsLoss"


    def build(self, input_dim: int, context_length: int = 0):
        from models.transformer import Transformer

        return Transformer(
            in_channels=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            seq_len=context_length if context_length is not None else 168,
        )

@dataclass
class LSTMConfig:
    name:            str       = "lstm"
    batch_size:      int       = 64
    hidden_size:     int       = 128
    num_layers:      int       = 2
    dropout:         float     = 0.3
    num_classes:     int       = 1
    pos_weight_cap:  float     = 10.0
    loss_fn:         str       = "BCEWithLogitsLoss"

    def build(self, input_dim: int):
        from models.LSTM import LSTMModel

        return LSTMModel(
            in_channels=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )


## HuggingFace configs cannot use @strict alongside @dataclass inheritance without
# breaking PretrainedConfig's internal __init__ chain. Instead, we use a plain
# @dataclass and call super().__init__() explicitly in __post_init__, passing all
# HF fields through. This ensures HF internals (attn implementation, id2label, etc.)
# are set up correctly while keeping the clean dataclass field definition style.
@dataclass
class PatchTSTConfig():
    """
    PatchTST config extending HuggingFace's PatchTSTConfig with our training fields
    and alias properties so the pipeline can treat all model configs uniformly.

    PatchTST splits a time series into fixed-length patches and processes them with
    a Transformer encoder — similar to how Vision Transformers treat image patches.

    HuggingFace uses num_attention_heads / num_hidden_layers internally. The nhead
    and num_layers properties below provide uniform naming across all model configs.
    num_classes is stored directly as a field rather than delegating to HuggingFace's
    num_labels, which requires id2label to be set.

    __post_init__ forwards all relevant fields to PretrainedConfig.__init__ to ensure
    HuggingFace internals (attention implementation, label mappings, etc.) are
    properly initialized before the model is built.
    """

    name:                str   = "patchtst"
    batch_size:          int   = 64
    context_length:      int   = 168
    patch_length:        int   = 16
    patch_stride:        int   = 8
    d_model:             int   = 128
    nhead:               int   = 4
    num_layers:          int   = 3
    ffn_dim:             int   = 256
    channel_attention:   bool  = True
    attention_dropout:   float = 0.1
    positional_dropout:  float = 0.1
    head_dropout:        float = 0.1
    pre_norm:            bool  = True
    norm_type:           Literal["batchnorm", "layernorm"] = "batchnorm"
    pos_weight_cap:      float = 10.0
    loss_fn:             str   = "BCEWithLogitsLoss"
    num_classes:         int   = 1

    def build(self, input_dim: int, context_length = None):
        from models.PatchTST import PatchTST

        from transformers import PatchTSTConfig as HF_PatchTSTConfig # We have to extend the HuggingFace config

        hf_config = HF_PatchTSTConfig(
            num_input_channels=input_dim,
            context_length=context_length or self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_stride,
            d_model=self.d_model,
            num_attention_heads=self.nhead,
            num_hidden_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            positional_dropout=self.positional_dropout,
            attention_dropout=self.attention_dropout,
            head_dropout=self.head_dropout,
            channel_attention=self.channel_attention,
            pre_norm=self.pre_norm,
            norm_type=self.norm_type,
            num_targets=self.num_classes,
        )
        return PatchTST(hf_config)

# ── Data configs ──────────────────────────────────────────────────────── #

@dataclass
class LeadCSVConfig: 
    name:         str   = "lead_csv"
    file_path:    Path  = Path("datasets/LEAD/train_features_clean.csv") #used for shared mode, ignored for local mode, should be the large dataset csv (sentinel-cleaned by scripts/clean_lead_features.py)
    data_dir:     Path  = Path("datasets/LEAD")
    file_pattern: str   = "data{client_index}.csv"
    target:       str   = "anomaly"
    task_name:    str   = "binary_classification"
    test_split:   float = 0.2
    seed:         int   = 42

    def build_handler(self, config=None):
        from data.lead_csv import LeadCSVHandler
        return LeadCSVHandler(config or CONFIG)

@dataclass
class PowerGridCSVConfig:
    name:         str   = "powergrid_csv"
    file_path:    Path  = Path("datasets/EPIC/Scenario_1/EpicLog_noisy.csv")
    clean_path:   Path  = Path("datasets/EPIC/Scenario_1/EpicLog_clean.csv")
    target:       str   = "marker"
    test_split:   float = 0.2
    normalize:    bool  = True
    noise_level:  float = 0.7
    seed:         int   = 42

    def build_handler(self, config=None):
        from data.powergrid_csv import PowerGridCSVHandler
        return PowerGridCSVHandler(config or CONFIG)

# ── Training config ───────────────────────────────────────────────────── #

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    local_epochs:  int   = 2
    patience:      int   = 5

# ── Federation config ─────────────────────────────────────────────────── #

@dataclass
class FederationConfig:
    partition_mode:    str   = "local" # local or shared
    num_rounds:        int   = 2
    num_clients:       int   = 1
    fraction_fit:      float = 1.0
    fraction_evaluate: float = 1.0
    proximal_mu:       float = 0.1

# ── Evaluation config ─────────────────────────────────────────────────── #

@dataclass
class EvaluationConfig:
    model_path:   Path | None  = None  # set in FedProxWithSave
    test_path:    Path         = Path("/app/datasets/LEAD/test_features_clean.csv")
    target:       str          = "anomaly"
    batch_size:   int          = 64
    threshold:    float        = 0.5   # decision threshold — override with best_thresh from training
    input_dim:    int          = 0     # set after data loading

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
# Change CONFIG to switch experiments. All fields have defaults so you only
# need to specify what differs from the defaults.

CONFIG = ExperimentConfig(
    model=CNNTransformerConfig(nhead=4, num_layers=1, batch_size=128, pos_weight_cap=100, dropout=0.1250780386573224, d_model=64),
    training=TrainingConfig(local_epochs=20, learning_rate=0.00016273524419282967, weight_decay=0.0001000950072852069),
    federation=FederationConfig(num_rounds=10, num_clients=2, proximal_mu=0.1, partition_mode="shared"),
)


"""
Examples:

FedProx with LSTM on PowerGrid dataset:
CONFIG = ExperimentConfig(
    model=ModelConfig(model=LSTMConfig(hidden_size=256, dropout=0.2)),
    data=PowerGridCSVConfig(),
    training=TrainingConfig(learning_rate=5e-4, local_epochs=1),
    federation=FederationConfig(num_rounds=10, num_clients=3, proximal_mu=0.1),
    evaluation=EvaluationConfig(threshold=0.3),
)

PatchTST on LEAD dataset:
CONFIG = ExperimentConfig(
    model=PatchTSTConfig(nhead=4, num_layers=3),
    data=LeadCSVConfig(batch_size=32),
    training=TrainingConfig(local_epochs=5, learning_rate=1e-5),
)

MLP baseline on LEAD dataset:
CONFIG = ExperimentConfig(
    model=MLPConfig(hidden_size=64, num_layers=3),
)
"""