# ── Active experiment ─────────────────────────────────────────────────── #
# Change CONFIG to switch experiments. All fields have defaults so you only
# need to specify what differs from the defaults.

from config import *

CONFIG = ExperimentConfig(
    model=CNNTransformerConfig(
        nhead=4,
        num_layers=2,
        batch_size=128,
        pos_weight_cap=50,
        dropout=0.2,
        d_model=64,
    ),
    training=TrainingConfig(
        local_epochs=5,
        learning_rate=0.00016273524419282967,
        weight_decay=0.0001000950072852069,
    ),
    federation=FederationConfig(
        num_rounds=50,
        num_clients=10,
        proximal_mu=0.1,
        partition_mode="shared",
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