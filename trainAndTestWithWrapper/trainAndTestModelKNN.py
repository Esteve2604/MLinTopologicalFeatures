from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndTrainModel

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
# sm = SVMSMOTE(sampling_strategy={0: 40, 1: 40}, m_neighbors=5, k_neighbors=4)
# X_train, y_train = sm.fit_resample(X_train,y_train)
 # Define the parameter grid for GridSearchCV

model = KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=15, p=2, weights='uniform')

selectFeaturesSFSAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['AnumberOfPointsH1', 'EentropyH1', 'EbettiAmplitudeH1', 'InumberOfPointsH1', 'IbettiAmplitudeH0', 'OentropyH2', 'OnumberOfPointsH0', 'OcomplexPolinomialCoef6H1', 'UentropyH0', 'UnumberOfPointsH1', 'UbettiAmplitudeH0', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef6H0', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef6H1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.70      0.67        10
#            1       0.67      0.60      0.63        10

#     accuracy                           0.65        20
#    macro avg       0.65      0.65      0.65        20
# weighted avg       0.65      0.65      0.65        20

# ___SFS____:
# Selected features: ['AbettiAmplitudeH0', 'EentropyH2', 'InumberOfPointsH1', 'InumberOfPointsH2', 'OnumberOfPointsH0', 'OnumberOfPointsH1', 'ObettiAmplitudeH2', 'OcomplexPolinomialCoef1H1', 'OcomplexPolinomialCoef6H1', 'OcomplexPolinomialCoef6H2', 'UnumberOfPointsH0', 'UnumberOfPointsH1', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UbettiAmplitudeH2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.55      0.60      0.57        10
#            1       0.56      0.50      0.53        10

#     accuracy                           0.55        20
#    macro avg       0.55      0.55      0.55        20
# weighted avg       0.55      0.55      0.55        20


# Confusion Matrix:
#  [[6 4]
#  [5 5]]

# Confusion Matrix:
#  [[7 3]
#  [4 6]]
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['AbettiAmplitudeH2', 'EnumberOfPointsH2', 'IbettiAmplitudeH0', 'IbettiAmplitudeH1', 'ObettiAmplitudeH0', 'OcomplexPolinomialCoef6H1', 'UentropyH2', 'UnumberOfPointsH2', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UcomplexPolinomialCoef6H0', 'UcomplexPolinomialCoef2H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef6H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.58      0.70      0.64        10
#            1       0.62      0.50      0.56        10

#     accuracy                           0.60        20
#    macro avg       0.60      0.60      0.60        20
# weighted avg       0.60      0.60      0.60        20


# Confusion Matrix:
#  [[7 3]
#  [5 5]]