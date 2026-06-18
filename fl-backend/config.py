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
        from models.CNNTransformer.CNNTransformer import CNNTransformer

        return CNNTransformer(
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
    context_length: int   = 168
    d_model:        int   = 128
    nhead:          int   = 4
    num_layers:     int   = 2
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        str   = "BCEWithLogitsLoss"


    def build(self, input_dim: int, context_length: int = 168):
        from models.CNNTransformer.Transformer import Transformer

        return Transformer(
            in_channels=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            seq_len=context_length,
        )
    
@dataclass
class CNNConfig:
    name:           str   = "cnn"
    batch_size:     int   = 64  
    d_model:        int   = 128
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int):
        from models.CNNTransformer.CNN import CNN

        return CNN(
            in_channels=input_dim,
            d_model=self.d_model,
            num_classes=self.num_classes,
            dropout=self.dropout,
        )

@dataclass
class LSTMConfig:
    name:              str   = "lstm"
    batch_size:        int   = 64
    context_length:    int   = 168
    lstm_units:        int   = 128
    dropout:           float = 0.3
    dimension_shuffle: bool  = False
    num_classes:       int   = 1
    pos_weight_cap:    float = 10.0
    loss_fn:           str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int, context_length: int = None):
        import torch.nn as nn
        from models.MLSTM_FCN.LSTM import LSTM
        from models.utils import Squeeze

        backbone = LSTM(
            input_dim=input_dim,
            num_timesteps=context_length or self.context_length,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            dimension_shuffle=self.dimension_shuffle,
        )
        return nn.Sequential(
            backbone,
            nn.Linear(self.lstm_units, self.num_classes),
            Squeeze(-1),
        )


@dataclass
class FCNConfig:
    name:           str   = "fcn"
    batch_size:     int   = 64
    use_se:         bool  = True
    se_reduction:   int   = 16
    num_classes:    int   = 1
    pos_weight_cap: float = 10.0
    loss_fn:        str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int):
        import torch.nn as nn
        from models.MLSTM_FCN.FCN import FCN
        from models.utils import Squeeze

        backbone = FCN(
            num_variables=input_dim,
            use_se=self.use_se,
            se_reduction=self.se_reduction,
        )
        return nn.Sequential(
            backbone,
            nn.Linear(FCN.OUTPUT_DIM, self.num_classes),
            Squeeze(-1),
        )


@dataclass
class MLSTMFCNConfig:
    name:              str   = "mlstm_fcn"
    batch_size:        int   = 64
    context_length:    int   = 168
    lstm_units:        int   = 8
    dropout:           float = 0.8
    dimension_shuffle: bool  = True
    se_reduction:      int   = 16
    num_layers:        int   = 1
    num_classes:       int   = 1
    pos_weight_cap:    float = 10.0
    loss_fn:           str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int, context_length: int = None):
        from models.MLSTM_FCN.MLSTM_FCN import MLSTM_FCN

        return MLSTM_FCN(
            input_dim=input_dim,
            num_timesteps=context_length or self.context_length,
            num_classes=self.num_classes,
            dimension_shuffle=self.dimension_shuffle,
            lstm_units=self.lstm_units,
            dropout=self.dropout,
            se_reduction=self.se_reduction,
            num_layers=self.num_layers,
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
    context_length:      int   = 168 # Has to mach the window_size used in data config
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
    is_encoder_decoder:  bool  = False
    share_embedding:     bool  = True
    pooling_type:        str   = "mean"
    pos_weight_cap:      float = 10.0
    loss_fn:             str   = "BCEWithLogitsLoss"
    num_classes:         int   = 1
    task:                str   = "single_label_classification"

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
            problem_type=self.task,
            is_encoder_decoder=self.is_encoder_decoder,
            share_embedding=self.share_embedding,
            pooling_type=self.pooling_type,
        )
        return PatchTST(hf_config)


@dataclass
class TimesNetConfig:
    name:           str   = "timesnet"
    batch_size:     int   = 64
    d_model:        int   = 128
    num_layers:     int   = 2
    top_k:          int   = 3
    d_ffn:          int   = 256
    n_kernels:      int   = 6
    dropout:        float = 0.3
    pos_weight_cap: float = 10.0
    num_classes:    int   = 1
    loss_fn:        str   = "BCEWithLogitsLoss"

    def build(self, input_dim: int, context_length: int = None):
        from models.TimesNet import TimesNetModel

        return TimesNetModel(
            n_steps=context_length if context_length is not None else 168,
            n_features=input_dim,
            n_classes=self.num_classes,
            n_layers=self.num_layers,
            top_k=self.top_k,
            d_model=self.d_model,
            d_ffn=self.d_ffn,
            n_kernels=self.n_kernels,
            dropout=self.dropout,
        )
    


# ── Data configs ──────────────────────────────────────────────────────── #

@dataclass
class LeadCSVConfig: 
    name:         str   = "lead_csv"
    file_path:    Path  = Path("datasets/LEAD/train_features_clean.csv") #used for shared mode, ignored for local mode, should be the large dataset csv (sentinel-cleaned by scripts/clean_lead_features.py)
    data_dir:     Path  = Path("datasets/LEAD")
    file_pattern: str   = "data{client_index}.csv"
    precomputed_dir: Path = Path("datasets/LEAD/windowed")
    precomputed_pattern: str = "client{client_index}.npz"
    use_precomputed_windows: bool = False
    target:       str   = "anomaly"
    task_name:    str   = "binary_classification"
    # 60/20/20 temporal split: train/validation/final evaluation.
    # test_split is kept only for backwards compatibility with old code.
    train_split:  float = 0.6
    val_split:    float = 0.2
    eval_split:   float = 0.2
    test_split:   float = 0.2
    seed:         int   = 42
    window_size:  int   = 168
    gap_hours:    int   = 73
    stride:       int   = 168
    use_undersampling: bool = False
    use_oversampling: bool = True
    undersampling_ratio: float = 1.0
    oversampling_method: str = "borderline_smote" #["none", "random_over", "smote", "borderline_smote"]
    oversampling_ratio: float = 0.1
    smote_k_neighbors: int = 2
    undersample_val: bool = False
    oversample_val: bool = False

    def build_handler(self, config):
        from data.lead_csv import LeadCSVHandler
        return LeadCSVHandler(config)

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

    def build_handler(self, config):
        from data.powergrid_csv import PowerGridCSVHandler
        return PowerGridCSVHandler(config)

@dataclass
class PowerConsumptionAnomalyConfig:
    name:         str   = "power_consumption_anomaly"
    file_path:    Path  = Path("")
    data_dir:     Path  = Path("datasets/PowerConsumptionAnomaly")
    file_pattern: str   = "*.csv"
    precomputed_dir: Path = Path("datasets/PowerConsumptionAnomaly/windowed")
    precomputed_pattern: str = "client{client_index}.npz"
    use_precomputed_windows: bool = False
    target:       str   = "label"
    # 60/20/20 temporal split: train/validation/final evaluation.
    # test_split is kept only for backwards compatibility with old code.
    train_split:  float = 0.6
    val_split:    float = 0.2
    eval_split:   float = 0.2
    test_split:   float = 0.2
    normalize:    bool  = True
    seed:         int   = 42
    window_size:  int   = 60
    gap_hours:    int   = 0
    stride:       int   = 10
    use_undersampling: bool = True
    use_oversampling: bool = False
    undersampling_ratio: float = 1.0
    oversampling_method: str = "none" # ["none", "random_over", "smote"]
    oversampling_ratio: float = 1.0
    smote_k_neighbors: int = 5
    undersample_val: bool = True
    oversample_val: bool = False

    def build_handler(self, config):
        from data.power_consumption_anomaly import PowerConsumptionAnomalyHandler
        return PowerConsumptionAnomalyHandler(config)

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
    num_rounds:        int   = 15
    num_clients:       int   = 1
    fraction_fit:      float = 1.0
    fraction_evaluate: float = 1.0
    proximal_mu:       float = 2.0
    serialize_gpu:     bool  = True
    serialize_gpu_evaluate: bool = True
    gpu_lock_path:     Path  = Path("datasets/.gpu.lock")

# ── Evaluation config ─────────────────────────────────────────────────── #

@dataclass
class EvaluationConfig:
    model_path:   Path | None  = None  # set in FedProxWithSave
    test_path:    Path         = Path("datasets/LEAD/train_features_clean.csv")
    target:       str          = "anomaly"
    batch_size:   int          = 64
    threshold:    float        = 0.5   # decision threshold — overridden by the trained `threshold` metric
    input_dim:    int          = 0     # set after data loading


# ── Optuna search-space configs ───────────────────────────────────────── #
# Each model has its own search-space dataclass so ranges can differ per
# architecture (e.g. PatchTST needs a smaller batch_size grid because
# channel-independence multiplies effective batch by num_channels).
#
# OptunaOptimizer reads the dataclass attached to ExperimentConfig.optuna
# and translates each Range/list into a trial.suggest_* call.

@dataclass
class FloatRange:
    low:  float
    high: float
    log:  bool = False

@dataclass
class IntRange:
    low:  int
    high: int
    log:  bool = False

@dataclass
class GeneralOptunaConfig:
    n_trials: int = 50
    epochs: int = 10
    n_jobs: int = 1
    study_name: str | None = "Optuna_study"

@dataclass
class ScoringConfig:
    """Optuna optimizes pr_auc_weight * pr_auc + f1_weight * f1.
    Set (1, 0) for pure PR-AUC, (0, 1) for pure F1, or any mix."""
    pr_auc_weight: float = 0.5
    f1_weight:     float = 0.5


@dataclass
class PrunerConfig:
    min_resource_fraction: float = 0.25
    reduction_factor: int = 3

@dataclass
class OptunaSearchConfig:
    """Search space shared by every model."""
    lr:             FloatRange = field(default_factory=lambda: FloatRange(3e-4, 3e-3, log=True))
    weight_decay:   FloatRange = field(default_factory=lambda: FloatRange(1e-4, 1e-2, log=True))
    num_layers:     IntRange   = field(default_factory=lambda: IntRange(1, 5))
    batch_size:     list       = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    pos_weight_cap: FloatRange = field(default_factory=lambda: FloatRange(1.0, 10.0, log=True))


@dataclass
class LSTMOptunaConfig(OptunaSearchConfig):
    lstm_units:        list       = field(default_factory=lambda: [64, 128, 256, 512])
    dropout:           FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    dimension_shuffle: list       = field(default_factory=lambda: [True, False])


@dataclass
class FCNOptunaConfig(OptunaSearchConfig):
    use_se:       list = field(default_factory=lambda: [True, False])
    se_reduction: list = field(default_factory=lambda: [4, 8, 16, 32])


@dataclass
class MLSTMFCNOptunaConfig(OptunaSearchConfig):
    lstm_units:        list       = field(default_factory=lambda: [8, 16, 32, 64, 128])
    dropout:           FloatRange = field(default_factory=lambda: FloatRange(0.3, 0.8))
    dimension_shuffle: list       = field(default_factory=lambda: [True, False])
    se_reduction:      list       = field(default_factory=lambda: [8, 16, 32])
    num_layers:        list       = field(default_factory=lambda: [1, 2, 3])


@dataclass
class TransformerOptunaConfig(OptunaSearchConfig):
    d_model:          list       = field(default_factory=lambda: [32, 64, 128, 256])
    nhead_candidates: list       = field(default_factory=lambda: [2, 4, 8, 16])
    dropout:          FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))


@dataclass
class CNNTransformerOptunaConfig(OptunaSearchConfig):
    d_model:          list       = field(default_factory=lambda: [32, 64, 128, 256])
    nhead_candidates: list       = field(default_factory=lambda: [2, 4, 8, 16])
    dropout:          FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))


@dataclass
class PatchTSTOptunaConfig(OptunaSearchConfig):
    # PatchTST channel-independence multiplies effective batch by num_channels,
    # so its batch_size grid is intentionally smaller than the shared default.
    batch_size:              list       = field(default_factory=lambda: [32, 64, 128])
    d_model:                 list       = field(default_factory=lambda: [32, 64, 128, 256])
    nhead_candidates:        list       = field(default_factory=lambda: [2, 4, 8, 16])
    # patch_length is filtered against context_length at trial time so it
    # always evenly divides the window.
    patch_length_candidates: list       = field(default_factory=lambda: [8, 16, 32])
    # patch_stride is not searched: it's fixed to patch_length // 2 (50% overlap,
    # the PatchTST paper default). Searching it would require a dynamic choice
    # set per trial, which Optuna's storage does not allow.
    ffn_dim:                 list       = field(default_factory=lambda: [32, 64, 128, 256])
    channel_attention:       list       = field(default_factory=lambda: [True, False])
    # PatchTST has no shared `dropout` — only per-component dropouts below.
    attention_dropout:       FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    positional_dropout:      FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    head_dropout:            FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))

@dataclass
class TimesNetOptunaConfig(OptunaSearchConfig):
    """Search space used by TimesNetConfig."""

    # TimesNet can be fairly memory-heavy, so keep batch sizes moderate.
    batch_size: list = field(default_factory=lambda: [32, 64, 128, 256])

    d_model: list    = field(default_factory=lambda: [64, 128, 256])
    top_k: list      = field(default_factory=lambda: [2, 3, 5])
    d_ffn: list      = field(default_factory=lambda: [128, 256, 512])
    n_kernels: list  = field(default_factory=lambda: [3, 6, 9])
    dropout: FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    pos_weight_cap: FloatRange = field(default_factory=lambda: FloatRange(1.0, 10.0, log=True))
    num_layers: IntRange = field(default_factory=lambda: IntRange(1, 5))
                                                                  
@dataclass
class CNNOptunaConfig(OptunaSearchConfig):
    d_model: list    = field(default_factory=lambda: [32, 64, 128, 256])
    dropout: FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    pos_weight_cap: FloatRange = field(default_factory=lambda: FloatRange(1.0, 10.0, log=True))
    batch_size: list = field(default_factory=lambda: [32, 64, 128])
    

# ── Interpretability config ─────────────────────────────────────────── #
@dataclass
class InterpretabilityConfig:
    # Number of steps in Integrated Gradients
    ig_steps: int = 50

    # Baseline type: "zero", "mean", or "sample"
    ig_baseline: str = "zero"

    # Whether to use probabilities instead of raw logits
    ig_use_probability: bool = False

    # Where to save IG outputs
    ig_output_path: Path = Path("plotting/saved_plots/integrated_gradients.npz")

# ── Top-level experiment config ───────────────────────────────────────── #

    

@dataclass
class ExperimentConfig:
    task:       BinaryClassificationConfig   = field(default_factory=BinaryClassificationConfig)
    model:      CNNConfig                    = field(default_factory=CNNConfig)
    data:       LeadCSVConfig                = field(default_factory=LeadCSVConfig)
    training:   TrainingConfig               = field(default_factory=TrainingConfig)
    federation: FederationConfig             = field(default_factory=FederationConfig)
    evaluation: EvaluationConfig             = field(default_factory=EvaluationConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)

    def __post_init__(self):
        if hasattr(self.model, "context_length"):
            self.model.context_length = self.data.window_size


@dataclass
class OptunaConfig:
    general: GeneralOptunaConfig = field(default_factory=GeneralOptunaConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)

    lstm: LSTMOptunaConfig = field(default_factory=LSTMOptunaConfig)
    transformer: TransformerOptunaConfig = field(default_factory=TransformerOptunaConfig)
    cnn_transformer: CNNTransformerOptunaConfig = field(default_factory=CNNTransformerOptunaConfig)
    patchtst: PatchTSTOptunaConfig = field(default_factory=PatchTSTOptunaConfig)
    timesnet: TimesNetOptunaConfig = field(default_factory=TimesNetOptunaConfig)
    fcn: FCNOptunaConfig = field(default_factory=FCNOptunaConfig)
    mlstm_fcn: MLSTMFCNOptunaConfig = field(default_factory=MLSTMFCNOptunaConfig)
    cnn: CNNOptunaConfig = field(default_factory=CNNOptunaConfig)


CONFIG = ExperimentConfig()

OPTUNA_CONFIG = OptunaConfig()
