import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import  mutual_info_classif
from kneed import KneeLocator
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")
df.drop("sampleName", axis=1, inplace=True)

y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)

scores = mutual_info_classif(x,y)

mi_series = pd.Series(scores, index=x.columns)
mi_sorted = mi_series.sort_values(ascending=False)
knee = KneeLocator(
    range(len(mi_sorted)),
    mi_sorted,
    curve='convex',
    direction='decreasing'
)
threshold = mi_sorted[knee.knee]
plt.figure(figsize=(10, 5))
plt.plot(mi_sorted)
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.scatter(knee.knee + 1, mi_sorted[knee.knee], color='red', zorder=5, s=100, label='Elbow (knee)')
plt.title("Mutual Information Scores per Feature")
plt.ylabel("MI Score")
plt.legend()
plt.tight_layout()
plt.show()

mask = scores > threshold

# Filter X
X_filtered = x.loc[:, mask]


# Optional: combine with target column
df_selected = X_filtered.copy()
df_selected['parkinson?'] = y  # optional, if you want to keep the label too

# Step 4: Save to CSV
df_selected.to_csv('selected_ThresholdfeaturesPDRed2.5EV8kHzStride2.csv', index=False)
