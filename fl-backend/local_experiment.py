# ── Active experiment ─────────────────────────────────────────────────── #
# Change CONFIG to switch experiments. All fields have defaults so you only
# need to specify what differs from the defaults.

from pathlib import Path
from config import *

DATASET = "lead"  # "power" or "lead"


def build_data_config(dataset: str):
    if dataset == "power":
        return PowerConsumptionAnomalyConfig(
            data_dir=Path("datasets/PowerConsumptionAnomaly"),
            file_pattern="*.csv",
            window_size=168,
            stride=168,
            gap_hours=0,
            use_undersampling = False,
            use_oversampling = False,
            undersampling_ratio = 5.0,
            oversampling_method = "none", # ["none", "random_over", "smote"]
            oversampling_ratio = 1.0,
            smote_k_neighbors = 5,
            undersample_val = False,
            oversample_val = False,
        )

    if dataset == "lead":
        return LeadCSVConfig(
            file_path=Path("datasets/LEAD/train_features_clean.csv"),
            window_size=168,
            stride=168,
            gap_hours=73,
            use_undersampling = False,
            use_oversampling = True,
            undersampling_ratio = 5.0,
            oversampling_method = "time_series_augment", # ["none", "random_over", "smote", "borderline_smote","time_series_augment"]
            oversampling_ratio = 1.0, #best so far 0.5
            smote_k_neighbors = 2,
            undersample_val = False,
            oversample_val =False,
            use_precomputed_windows = True,
        )

    raise ValueError(f"Unknown dataset: {dataset}. Use 'power' or 'lead'.")


CONFIG = ExperimentConfig(
    
    #model=PatchTSTConfig(
      #  batch_size   = 64,
      #  context_length   = 168, # Has to mach the window_size used in data config
     #   patch_length   = 28,
     #   patch_stride   = 8,
     #   d_model   = 64,
     #   nhead   = 4,
     #   num_layers   = 3,
      #  ffn_dim   = 128,
     #   channel_attention  = True,
       # attention_dropout = 0.08533492544875493,
        #positional_dropout =0.2304421861914936,
        #head_dropout = 0.19012638100200221,
        #pre_norm  = True,
        #pos_weight_cap = 2.572223644629402
#,

    #),
    model = CNNTransformerConfig(
        batch_size   = 16,
        d_model   = 128,
        nhead  = 8,
        num_layers  = 2,
        dropout = 0.14,
        pos_weight_cap = 1.66,
    ),
    data=build_data_config(DATASET),
    training=TrainingConfig(
        local_epochs=3,
        learning_rate=  0.00080,
        weight_decay=0.00018,
        patience=10,
    ),
    federation=FederationConfig(
        num_rounds=1,
        num_clients=5,
        proximal_mu=0.0,
        partition_mode="local",
    ),
)

"""
Examples:

CONFIG = ExperimentConfig(
    model=TimesNetConfig(num_layers=3,d_model=128,top_k=3,d_ffn=256,n_kernels=6,dropout=0.3,num_classes=1,loss_fn="BCEWithLogitsLoss",),
    training=TrainingConfig(local_epochs=1, learning_rate=1e-4),
    federation=FederationConfig(num_rounds=2,num_clients=1, proximal_mu=0.1,partition_mode="local",),
)
    
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
