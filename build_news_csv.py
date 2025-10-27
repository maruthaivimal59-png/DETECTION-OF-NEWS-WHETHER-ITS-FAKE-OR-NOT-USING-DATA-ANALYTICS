import pandas as pd


df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")


df_true["text"] = df_true["title"].fillna("") + ". " + df_true["text"].fillna("")
df_fake["text"] = df_fake["title"].fillna("") + ". " + df_fake["text"].fillna("")

df_true["label"] = "REAL"
df_fake["label"] = "FAKE"

df = pd.concat([df_true[["text", "label"]], df_fake[["text", "label"]]], ignore_index=True)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

df.to_csv("news.csv", index=False)
print(" news.csv created with", len(df), "rows.")
