CONFIG = {
    "model": {
        "name":"cnn", #input model type here
        "hidden_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "data":{
        "name":"SWAT",#input dataset name here
        "batch_size": 32,
        "num_clients": 3,
        "sequence_length": 50,
        "num_features": 51,
        "num_classes": 2,
        "samples_per_client": 1000,
    },
    "training":{
        "learning_rate": 0.001,
        "local_epochs": 5,
    },
    "federation":{
        "num_rounds": 10,
        "fraction_train":1.0,
        "fraction_eval":1.0,
    },
}