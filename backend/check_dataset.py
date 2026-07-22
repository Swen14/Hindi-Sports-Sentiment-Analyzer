import pandas as pd

train = pd.read_csv("../dataset/hindi_sentiment_analysis_dedup.csv")

print(train.head())

print("\nColumns:")
print(train.columns)

print("\nShape:")
print(train.shape)