import pandas as pd
import numpy as np          
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
ordenc = OrdinalEncoder(categories=[3,2,1])
df = pd.read_csv("titanic/train.csv")
feature_names = ["Pclass","Sex","SibSp","Parch"]
features = pd.get_dummies(df[feature_names])
data_scaled = features.copy()
for column in data_scaled.columns:
    data_scaled[column] = (data_scaled[column] - data_scaled[column].mean()) / data_scaled[column].std(ddof=0)

print(data_scaled)