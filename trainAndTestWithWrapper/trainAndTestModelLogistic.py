# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndRFECVAndTrainModel

N_JOBS = 2
N_FEATURES_TO_SELECT = 15
CV = 2
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")
# df = pd.read_csv("./data/ufeaturesPDRed2.5EV8kHzStride2.csv")
df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv")
df.drop("sampleName", axis=1, inplace=True)

y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)

# Create a ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.columns.to_numpy()),
    ])




X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )

# Resample the training dataset to improve predictions
# sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)
 
model = LogisticRegression(random_state=42, C=1, class_weight=None, l1_ratio=0.0, max_iter=1250, penalty='elasticnet', solver='saga')

selectFeaturesSFSAndRFECVAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)

# ___RFECV____:

# Nº of features selected: 8
# Selected features: ['AentropyH0', 'AlandscapeAmplitudeH2', 'AnumberOfPointsH0', 'AbettiAmplitudeH2', 'AcomplexPolinomialCoef6H0', 'EentropyH2', 'EnumberOfPointsH0', 'UbettiAmplitudeH1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.90      0.75        10
#            1       0.83      0.50      0.62        10

#     accuracy                           0.70        20
#    macro avg       0.74      0.70      0.69        20
# weighted avg       0.74      0.70      0.69        20


# Confusion Matrix:
#  [[9 1]
#  [5 5]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['AentropyH0', 'AcomplexPolinomialCoef6H0', 'EbettiAmplitudeH2', 'IcomplexPolinomialCoef7H1', 'IcomplexPolinomialCoef9H2', 'ObettiAmplitudeH2', 'OcomplexPolinomialCoef8H2', 'UlandscapeAmplitudeH0', 'UlandscapeAmplitudeH2', 'UnumberOfPointsH0', 'UbettiAmplitudeH1', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.90      0.72        10
#            1       0.80      0.40      0.53        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[9 1]
#  [6 4]]
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___RFECV____:

# Nº of features selected: 3
# Selected features: ['AbettiAmplitudeH2', 'OlandscapeAmplitudeH2', 'UbettiAmplitudeH1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.90      0.72        10
#            1       0.80      0.40      0.53        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[9 1]
#  [6 4]]
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['AentropyH1', 'AcomplexPolinomialCoef6H0', 'IcomplexPolinomialCoef6H2', 'ObettiAmplitudeH2', 'UnumberOfPointsH2', 'UbettiAmplitudeH1', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.60      0.60      0.60        10
#            1       0.60      0.60      0.60        10

#     accuracy                           0.60        20
#    macro avg       0.60      0.60      0.60        20
# weighted avg       0.60      0.60      0.60        20


# Confusion Matrix:
#  [[6 4]
#  [4 6]]