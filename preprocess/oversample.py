import pandas as pd
from imblearn.over_sampling import SVMSMOTE

df = pd.read_csv("./data/features.csv")
df.drop("sampleName", axis=1, inplace=True)
y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)


sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
X_res, y_res = sm.fit_resample(X_train,y_train)

X_res["parkinson?"] = y_res

X_res.to_csv("./data/featuresOverSampled100.csv",index=False)
