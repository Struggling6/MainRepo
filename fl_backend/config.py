CONFIG = {
    "task": {
        "name": "anomaly_detection",
    },

    "model": {
        "name": "unsupcnntrans", 
        "hidden_dim": 64,
        "dropout": 0.2,
        "in_channels": 1,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
    },

    "data": {
        "name": "epic_csv", 
        "file_path": "datasets/EPIC/Scenario_1/EpicLog_noisy.csv", 
        "clean_path": "datasets/EPIC/Scenario_1/EpicLog_Scenario 1_19_Oct_2018_14_44.csv", 
        "label_column": "marker",
        "batch_size": 32,
        "num_clients": 10,
        "test_split": 0.2,
        "normalize": True,
        "seed": 42,
        "noise_level": 0.7, 
    },

    "training": {
        "learning_rate": 1e-4,
        "local_epochs": 5,
    },

    "federation": {
        "num_rounds": 10,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
    },
}