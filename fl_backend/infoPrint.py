import pandas as pd

df = pd.read_csv("datasets/LEAD/train_features.csv")

print(df["anomaly"].value_counts())
print(df.groupby("building_id").size().describe())
print(df.groupby("building_id")["anomaly"].sum().describe())
print((df.groupby("building_id")["anomaly"].sum() == 0).sum())

