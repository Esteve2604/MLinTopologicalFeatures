from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
from imblearn.over_sampling import SVMSMOTE

CV = 5

# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")
# df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
# # df = pd.read_csv("./data/univariate/selected_ThresholdAfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdEfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdIfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdOfeaturesPDRed13EV44,1kHzStride2.csv")
df = pd.read_csv("./data/univariate/selected_ThresholdUfeaturesPDRed13EV44,1kHzStride2.csv")

dfA = pd.read_csv("./data/afeaturesPDRed13EV41kHzStride2.csv") 
dfE = pd.read_csv("./data/efeaturesPDRed13EV41kHzStride2.csv") 
dfI = pd.read_csv("./data/ifeaturesPDRed13EV41kHzStride2.csv") 
dfO = pd.read_csv("./data/ofeaturesPDRed13EV41kHzStride2.csv") 
dfU = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") 

yA = dfA["parkinson?"]
xA = dfA.drop("parkinson?", axis=1)
X_trainA, X_testA, y_trainA, y_testA = train_test_split(
        xA, yA, test_size=0.2, stratify=yA
    )

# Eliminar las filas con esos IDs
dfE_test = dfE[~dfE['sampleName'].isin(xA['sampleName'])]
dfE_train = dfE[dfE['sampleName'].isin(xA['sampleName'])]
dfE_train.drop("sampleName", axis=1, inplace=True)
dfE_test.drop("sampleName", axis=1, inplace=True)
X_trainE = dfE_train.drop("parkinson?", axis=1)
y_trainE = dfE_train["parkinson?"]
X_testE = dfE_test.drop("parkinson?", axis=1)
y_testE = dfE_test["parkinson?"]

dfI_test = dfI[~dfI['sampleName'].isin(xA['sampleName'])]
dfI_train = dfI[dfI['sampleName'].isin(xA['sampleName'])]
dfI_train.drop("sampleName", axis=1, inplace=True)
dfI_test.drop("sampleName", axis=1, inplace=True)
X_trainI = dfI_train.drop("parkinson?", axis=1)
y_trainI = dfI_train["parkinson?"]
X_testI = dfI_test.drop("parkinson?", axis=1)
y_testI = dfI_test["parkinson?"]

dfO_test = dfO[~dfO['sampleName'].isin(xA['sampleName'])]
dfO_train = dfO[dfO['sampleName'].isin(xA['sampleName'])]
dfO_train.drop("sampleName", axis=1, inplace=True)
dfO_test.drop("sampleName", axis=1, inplace=True)
X_trainO = dfO_train.drop("parkinson?", axis=1)
y_trainO = dfO_train["parkinson?"]
X_testO = dfO_test.drop("parkinson?", axis=1)
y_testO = dfO_test["parkinson?"]

dfU_test = dfU[~dfU['sampleName'].isin(xA['sampleName'])]
dfU_train = dfU[dfU['sampleName'].isin(xA['sampleName'])]
dfU_train.drop("sampleName", axis=1, inplace=True)
dfU_test.drop("sampleName", axis=1, inplace=True)
X_trainU = dfU_train.drop("parkinson?", axis=1)
y_trainU = dfU_train["parkinson?"]
X_testU = dfU_test.drop("parkinson?", axis=1)
y_testU = dfU_test["parkinson?"]

X_trainA.drop("sampleName", axis=1, inplace=True)
X_testA.drop("sampleName", axis=1, inplace=True)


# Create a ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.columns.to_numpy()),
    ])





class CustomVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators  # list of (name, estimator, X_train, y_train)

    def fit(self, X=None, y=None):  # Ignore global X/y
        for name, model, X_train, y_train in self.estimators:
            model.fit(X_train, y_train)
        return self

    def predict(self, X):
        predictions = np.asarray([model.predict(X) for _, model, _, _ in self.estimators])
        # Majority vote
        maj_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return maj_vote