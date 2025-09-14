import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
SIMILARITY_CSV = "data/similarity_score_matrix.csv"
N_PLAYERS = None  # set to number of rows in mydata.csv if you need trimming
# ---------------------------

# Load similarity matrix
sim_df = pd.read_csv(SIMILARITY_CSV, encoding="utf8", delimiter=';', decimal=',', header=None)

# Trim to N_PLAYERS if specified
if N_PLAYERS is not None:
    sim_df = sim_df.iloc[:, :N_PLAYERS]

arr = sim_df.values.astype(float)

# Extract upper triangle (exclude diagonal)
scores = arr[np.triu_indices(arr.shape[0], k=1)]

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(scores, bins=40, range=(0, 1), color="skyblue", edgecolor="black")
plt.title("Distribution of similarity scores")
plt.xlabel("Similarity score")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
