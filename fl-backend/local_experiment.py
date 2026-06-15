from pathlib import Path
from config import *

DATASET = "lead"  # "power" or "lead"


def build_data_config(dataset: str):
    if dataset == "power":
        return PowerConsumptionAnomalyConfig(
            data_dir=Path("datasets/PowerConsumptionAnomaly/partitions"),
            file_pattern="data{client_index}.csv",
            precomputed_dir=Path("datasets/PowerConsumptionAnomaly/windowed"),
            precomputed_pattern="client{client_index}.npz",
            window_size=168,
            stride=168,
            gap_hours=0,
            use_undersampling = True,
            use_oversampling = True,
            undersampling_ratio = 1.0,
            oversampling_method = "time_series_augment", # ["none", "random_over", "smote"]
            oversampling_ratio = 1.0,
            smote_k_neighbors = 5,
            undersample_val = True,
            oversample_val = True,
            use_precomputed_windows = True,
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
            oversample_val =True,
            use_precomputed_windows = False,
        )

    raise ValueError(f"Unknown dataset: {dataset}. Use 'power' or 'lead'.")

CONFIG = ExperimentConfig(
    model = CNNConfig(
        batch_size   = 32,
        dropout = 0.37490607623107347,
        pos_weight_cap = 2.1471864748084784,
        d_model=64,
    ),
    data=build_data_config(DATASET),
    training=TrainingConfig(
        local_epochs=1,
        learning_rate=  0.0010563268303650562,
        weight_decay= 0.00019180167203803068,
        patience=10,
    ),
    federation=FederationConfig(
        num_rounds=1,
        num_clients=1,
        proximal_mu=0.0001,
        partition_mode="local",
    ),
   # evaluation=EvaluationConfig(
    #    test_path =Path("datasets/PowerConsumptionAnomaly/Power-Consumption-Anomaly-Dataset-main/eval"),
    #    target = "label",
     #   batch_size = 64, 
    #),
)
