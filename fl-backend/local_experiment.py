from pathlib import Path
from config import *

DATASET = "power"  # "power" or "lead"


def build_data_config(dataset: str):
    if dataset == "power":
        return PowerConsumptionAnomalyConfig(
            data_dir=Path("datasets/PowerConsumptionAnomaly/partitions"),
            file_pattern="data{client_index}.csv",
            precomputed_dir=Path("datasets/PowerConsumptionAnomaly/windowed"),
            precomputed_pattern="client{client_index}.npz",
            use_precomputed_windows=True,
            window_size=168,
            stride=168,
            gap_hours=0,
            use_undersampling = False,
            use_oversampling = True,
            undersampling_ratio = 1.0,
            oversampling_method = "time_series_augment", # ["none", "random_over", "smote"]
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
        local_epochs=1,
        learning_rate=  0.00080,
        weight_decay=0.00018,
        patience=10,
    ),
    federation=FederationConfig(
        num_rounds=1,
        num_clients=2,
        proximal_mu=0.0,
        partition_mode="local",
    ),
    evaluation=EvaluationConfig(
        test_path =Path("datasets/PowerConsumptionAnomaly/Power-Consumption-Anomaly-Dataset-main/eval"),
        target = "label",
        batch_size = 64, 
    ),
)
