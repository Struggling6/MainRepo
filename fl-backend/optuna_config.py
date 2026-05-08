from dataclasses import dataclass, field


@dataclass
class FloatRange:
    low: float
    high: float
    log: bool = False


@dataclass
class IntRange:
    low: int
    high: int
    log: bool = False


@dataclass
class GeneralOptunaConfig:
    n_trials: int = 150
    epochs: int = 10
    n_jobs: int = 4
    study_name: str | None = "general cnn+trans study"


@dataclass
class ScoringConfig:
    pr_auc_weight: float = 0.5
    f1_weight: float = 0.5
    report_metric: str = "score"  # "score", "pr_auc", or "f1"


@dataclass
class PrunerConfig:
    min_resource_fraction: float = 0.25
    reduction_factor: int = 3


@dataclass
class OptunaSearchConfig:
    """Search space shared by every model."""

    lr: FloatRange = field(default_factory=lambda: FloatRange(3e-6, 3e-3, log=True))
    weight_decay: FloatRange = field(default_factory=lambda: FloatRange(1e-6, 1e-2, log=True))
    num_layers: IntRange = field(default_factory=lambda: IntRange(1, 6))
    batch_size: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    pos_weight_cap: FloatRange = field(default_factory=lambda: FloatRange(1.0, 90.0, log=True))
    dropout: FloatRange = field(default_factory=lambda: FloatRange(0.15, 0.55))


@dataclass
class LSTMOptunaConfig(OptunaSearchConfig):
    """Search space used by LSTMConfig."""

    hidden_size: list[int] = field(default_factory=lambda: [64, 128, 256, 512])


@dataclass
class TransformerOptunaConfig(OptunaSearchConfig):
    """Search space used by TransformerConfig."""

    d_model: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    nhead_candidates: list[int] = field(default_factory=lambda: [2, 4, 8, 16])


@dataclass
class CNNTransformerOptunaConfig(OptunaSearchConfig):
    """Search space used by CNNTransformerConfig."""

    d_model: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    nhead_candidates: list[int] = field(default_factory=lambda: [2, 4, 8, 16])


@dataclass
class PatchTSTOptunaConfig(OptunaSearchConfig):
    """Search space used by PatchTSTConfig."""

    # PatchTST channel-independence multiplies effective batch by num_channels,
    # so its batch_size grid is intentionally smaller than the shared default.
    batch_size: list[int] = field(default_factory=lambda: [32, 64, 128])

    d_model: list[int] = field(default_factory=lambda: [32, 64, 128])
    nhead_candidates: list[int] = field(default_factory=lambda: [2, 4, 8, 16])

    # patch_length is filtered against context_length at trial time so it
    # always evenly divides the window.
    patch_length_candidates: list[int] = field(default_factory=lambda: [4, 8, 16, 32])

    ffn_dim: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    channel_attention: list[bool] = field(default_factory=lambda: [True, False])

    attention_dropout: FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    positional_dropout: FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))
    head_dropout: FloatRange = field(default_factory=lambda: FloatRange(0.1, 0.4))


@dataclass
class OptunaConfig:
    general: GeneralOptunaConfig = field(default_factory=GeneralOptunaConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    pruner: PrunerConfig = field(default_factory=PrunerConfig)

    lstm: LSTMOptunaConfig = field(default_factory=LSTMOptunaConfig)
    transformer: TransformerOptunaConfig = field(default_factory=TransformerOptunaConfig)
    cnn_transformer: CNNTransformerOptunaConfig = field(default_factory=CNNTransformerOptunaConfig)
    patchtst: PatchTSTOptunaConfig = field(default_factory=PatchTSTOptunaConfig)


OPTUNA_CONFIG = OptunaConfig()