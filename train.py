import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("../data/defence_data.csv")

le = LabelEncoder()
df["threat"] = le.fit_transform(df["threat"])

X = df[["temperature", "movement", "signal_strength"]]
y = df["threat"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump((model, le), open("../models/model.pkl", "wb"))

print("Model trained and saved!")
