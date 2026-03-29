import pandas as pd
import torch
import numpy as np
from pathlib import Path

# Gaussian noise på numeriske kolonner
def add_gaussian_noise(df, columns, noise_level):
    df_noisy = df.copy()
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            noise = np.random.normal(0, noise_level * df[col].std(), len(df))
            df_noisy[col] = df[col] + noise
    return df_noisy

##TILFØJ SALT AND PEPPER INJECTION TIL BOOLEAN VÆRDIER !!!!!!

def inject_noise(input_path, output_path, noise_level):
    df = pd.read_csv(input_path)
    total_rows = len(df)
    threshold_row = int(total_rows * 0.8) 

    df_noisy = df.copy()

    # henter nueriske columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    #Tifløj kun noise i se sidste 20% af rækkerne
    noise_data = add_gaussian_noise(df.iloc[threshold_row:], numeric_cols, noise_level)
    df_noisy.iloc[threshold_row:] = noise_data.values

    df_noisy.to_csv(output_path, index=False)

    print(f"Noisy data gemt til: {output_path}")
    print(f"Total rows: {total_rows}")
    print(f"Noise injected in rows: {threshold_row} to {total_rows} ({(total_rows - threshold_row)} rows)")



# Sammenligning af dataset med nye 
#df_orig = pd.read_csv(path)  # Read original from same path
#df_noisy_check = pd.read_csv(output_path)

# Calculate difference for numeric columns
#ØHH find ud af hvordan man gør det girl det guirl