CONFIG = {
    "task": {
        "name": "anomaly_detection",
    },

    "model": {
        "name": "unsupcnntrans", 
        "hidden_dim": 0,
        "dropout": 0.2,
        "in_channels": 1,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
    },

    "data": {
        "name": "lead_csv", 
        "file_path": "datasets/LEAD/train_features.csv", 
        "label_column": "anomaly",
        "batch_size": 32,
        "num_clients": 10,
        "test_split": 0.2,
        "normalize": False,
        "seed": 42,
        "noise_level": 0.0, 
    },

    "training": {
        "learning_rate": 1e-4,
        "local_epochs": 10,
    },

    "federation": {
        "num_rounds": 10,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
    },
}