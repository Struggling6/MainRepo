CONFIG = {
    "task": {
        "name": "classification",
    },

    "model": {
        "name": "mlp",
        "hidden_dim": 64,
        "dropout": 0.2,
    },

    "data": {
        "name": "powergrid_csv",
        "file_path": "datasets/data1.csv",
        "label_column": "marker",
        "batch_size": 32,
        "num_clients": 10,
        "test_split": 0.2,
        "normalize": True,
        "seed": 42,
    },

    "training": {
        "learning_rate": 0.001,
        "local_epochs": 5,
    },

    "federation": {
        "num_rounds": 10,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
    },
}