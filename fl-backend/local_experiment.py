# ── Active experiment ─────────────────────────────────────────────────── #
# Change CONFIG to switch experiments. All fields have defaults so you only
# need to specify what differs from the defaults.

from pathlib import Path
from config import *

CONFIG = ExperimentConfig(
    
    model = CNNTransformerConfig(
       batch_size   = 64,
        d_model   = 128,
        nhead   = 4,
        num_layers   = 2,
        dropout = 0.3,
        pos_weight_cap = 10.0,

    ),

    #model = TimesNetConfig(
       # batch_size   = 64,
      #  d_model   = 128,
       # num_layers   = 2,
      #  top_k   = 3,
      #  d_ffn   = 256,
      #  n_kernels   = 6,
       # dropout = 0.3,
       # pos_weight_cap = 10.0,
       # num_classes   = 1,
    #),
    data=LeadCSVConfig(
        file_path=Path("datasets/LEAD/train_features_clean.csv"),
        window_size   = 168,
        gap_hours   = 73,
        stride  = 168,
        use_undersampling = True,
        use_oversampling = True,
        undersampling_ratio = 5.0,
        oversampling_method = "smote", #["none", "random_over", "smote"]
        oversampling_ratio = 1.0,
        smote_k_neighbors = 5,
        undersample_val = True,
        oversample_val = True,
    ),
    training=TrainingConfig(
        local_epochs=30,
        learning_rate=0.0009,
        weight_decay=0.0034,
        patience=10,
    ),
    federation=FederationConfig(
        num_rounds=1,
        num_clients=1,
        proximal_mu=0,
        partition_mode="optuna",
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