import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from kneed import KneeLocator
df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

df.drop("sampleName", axis=1, inplace=True)

y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)

selector = SelectKBest(mutual_info_classif, k=15).fit(x, y)


# Step 2: Get selected feature names
selected_features = x.columns[selector.get_support()]

# Step 3: Filter original DataFrame
X_selected = x[selected_features]

# Optional: combine with target column
df_selected = X_selected.copy()
df_selected['parkinson?'] = y  # optional, if you want to keep the label too

# Step 4: Save to CSV
df_selected.to_csv('selected_15featuresPDRed2.5EV8kHzStride2.csv', index=False)


mi_series = pd.Series(selector.scores_, index=x.columns)
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
