CONFIG = {
    "task": {
        "name": "anomaly_detection", #Eller hvad er det vi kalder det her???
    },

    "model": {
        "name": "cnn_transformer",
        "in_channels": 1,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },

    "data": {
        "name": "epic_csv",
        "file_path": "datasets/EPIC/Scenario_1/EpicLog_noisy.csv", #Tænker denne skal slettes senere så den automatisk laver en men lige ny giver den error hvis den laver en ny fil og den så er tom. 
        "clean_path": "datasets/EPIC/Scenario_1/EpicLog_Scenario 1_19_Oct_2018_14_44.csv", #
        "label_column": " ",
        "batch_size": 32,
        "num_clients": 10,
        "test_split": 0.2,
        "noise_level": 0.7, #
        "normalize": True,
        "seed": 42,
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