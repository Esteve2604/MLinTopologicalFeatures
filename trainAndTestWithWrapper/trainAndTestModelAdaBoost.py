from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier 
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndRFECVAndTrainModel
CV = 2
N_JOBS = 2
N_FEATURES_TO_SELECT = 15
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
# datasetPath = "./data/afeaturesPDRed13EV41kHzStride2.csv"
datasetPath = "./data/efeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ifeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ofeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ufeaturesPDRed13EV41kHzStride2.csv"

df = pd.read_csv(datasetPath)
print("El dataset utilizado es: " + datasetPath)

df.drop("sampleName", axis=1, inplace=True)

y = df["parkinson?"]
x = df.drop("parkinson?", axis=1)

# Create a ColumnTransformer to apply different preprocessing to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), x.columns.to_numpy())
        # ('scale', MaxAbsScaler(), x.columns.to_numpy())
    ])


X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
# Resample the training dataset to improve predictions

# Best params without sampling
# {'estimator': DecisionTreeClassifier(max_depth=1), 'learning_rate': 0.05, 'n_estimators': 75}
# Best Params with sampling
# {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__estimator': DecisionTreeClassifier(max_depth=7), 'trainingModel__learning_rate': 0.1, 'trainingModel__n_estimators': 150}
sm = SVMSMOTE(sampling_strategy={0: 40, 1: 40}, m_neighbors=4, k_neighbors=3)
X_train, y_train = sm.fit_resample(X_train,y_train)
model = AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(max_depth=7), learning_rate=0.1, n_estimators=150)

# model = AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(max_depth=1), learning_rate=0.05, n_estimators=75)

selectFeaturesSFSAndRFECVAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)

# NO RESAMPLING 0.2
# Nº of features selected: 9
# Selected features: ['AentropyH0', 'AentropyH2', 'AnumberOfPointsH2', 'UcomplexPolinomialCoef2H0', 'UcomplexPolinomialCoef4H0', 'UcomplexPolinomialCoef8H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.83      0.50      0.62        10
#            1       0.64      0.90      0.75        10

#     accuracy                           0.70        20
#    macro avg       0.74      0.70      0.69        20
# weighted avg       0.74      0.70      0.69        20


# Confusion Matrix:
#  [[5 5]
#  [1 9]]
 
# Nº of features selected: 1
# Selected features: ['UcomplexPolinomialCoef6H0']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.86      0.60      0.71        10
#            1       0.69      0.90      0.78        10

#     accuracy                           0.75        20
#    macro avg       0.77      0.75      0.74        20
# weighted avg       0.77      0.75      0.74        20


# Confusion Matrix:
#  [[6 4]
#  [1 9]]

# Nº of features selected: 17
# Selected features: ['AentropyH0', 'AnumberOfPointsH2', 'AbettiAmplitudeH2', 'OentropyH1', 'OnumberOfPointsH1', 'ObettiAmplitudeH2', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.30      0.43        10
#            1       0.56      0.90      0.69        10

#     accuracy                           0.60        20
#    macro avg       0.66      0.60      0.56        20
# weighted avg       0.66      0.60      0.56        20


# Confusion Matrix:
#  [[3 7]
#  [1 9]]
# Nº of features selected: 10
# Selected features: ['AentropyH0', 'AlandscapeAmplitudeH0', 'AnumberOfPointsH2', 'IbettiAmplitudeH2', 'ObettiAmplitudeH0', 'ObettiAmplitudeH2', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef9H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.50      0.67        10
#            1       0.67      1.00      0.80        10

#     accuracy                           0.75        20
#    macro avg       0.83      0.75      0.73        20
# weighted avg       0.83      0.75      0.73        20


# Confusion Matrix:
#  [[ 5  5]
#  [ 0 10]]

# Nº of features selected: 6
# Selected features: ['AentropyH0', 'AlandscapeAmplitudeH0', 'AnumberOfPointsH2', 'AbettiAmplitudeH2', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef10H0']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       1.00      0.40      0.57        10
#            1       0.62      1.00      0.77        10

#     accuracy                           0.70        20
#    macro avg       0.81      0.70      0.67        20
# weighted avg       0.81      0.70      0.67        20


# Confusion Matrix:
#  [[ 4  6]
#  [ 0 10]]

# Nº of features selected: 6
# Selected features: ['AentropyH0', 'AnumberOfPointsH2', 'UnumberOfPointsH1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.60      0.67        10
#            1       0.67      0.80      0.73        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[6 4]
#  [2 8]]
# RESAMPLE 0.2
# Nº of features selected: 4
# Selected features: ['AnumberOfPointsH2', 'IbettiAmplitudeH2', 'OnumberOfPointsH1', 'ObettiAmplitudeH0']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.78      0.70      0.74        10
#            1       0.73      0.80      0.76        10

#     accuracy                           0.75        20
#    macro avg       0.75      0.75      0.75        20
# weighted avg       0.75      0.75      0.75        20


# Confusion Matrix:
#  [[7 3]
#  [2 8]]

# Nº of features selected: 3
# Selected features: ['AnumberOfPointsH1', 'IbettiAmplitudeH2', 'ObettiAmplitudeH0']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.73      0.80      0.76        10
#            1       0.78      0.70      0.74        10

#     accuracy                           0.75        20
#    macro avg       0.75      0.75      0.75        20
# weighted avg       0.75      0.75      0.75        20


# Confusion Matrix:
#  [[8 2]
#  [3 7]]

#NO RESAMPLE
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['EcomplexPolinomialCoef10H2', 'IcomplexPolinomialCoef5H1', 'IcomplexPolinomialCoef5H2', 'OlandscapeAmplitudeH1', 'OcomplexPolinomialCoef6H0', 'OcomplexPolinomialCoef7H2', 'OcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef1H0', 'UcomplexPolinomialCoef3H0', 'UcomplexPolinomialCoef5H0', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:


# Classification Report:
#                precision    recall  f1-score   support

#            0       0.55      0.80      0.65        15
#            1       0.62      0.33      0.43        15

#     accuracy                           0.57        30
#    macro avg       0.59      0.57      0.54        30
# weighted avg       0.59      0.57      0.54        30


# Confusion Matrix:
#  [[12  3]
#  [10  5]]

# NO RESAMPLE 0.2
# ___RFECV____:

# Nº of features selected: 126
# Selected features: ['AentropyH0', 'AlandscapeAmplitudeH0', 'EcomplexPolinomialCoef5H2', 'IlandscapeAmplitudeH0', 'IlandscapeAmplitudeH1', 'IlandscapeAmplitudeH2', 'InumberOfPointsH0', 'InumberOfPointsH1', 'InumberOfPointsH2', 'IbettiAmplitudeH0', 'IbettiAmplitudeH1', 'IbettiAmplitudeH2', 'IcomplexPolinomialCoef1H0', 'IcomplexPolinomialCoef2H0', 'IcomplexPolinomialCoef3H0', 'IcomplexPolinomialCoef4H0', 'IcomplexPolinomialCoef5H0', 'IcomplexPolinomialCoef6H0', 'IcomplexPolinomialCoef7H0', 'IcomplexPolinomialCoef8H0', 'IcomplexPolinomialCoef9H0', 'IcomplexPolinomialCoef10H0', 'IcomplexPolinomialCoef1H1', 'IcomplexPolinomialCoef2H1', 'IcomplexPolinomialCoef3H1', 'IcomplexPolinomialCoef4H1', 'IcomplexPolinomialCoef5H1', 'IcomplexPolinomialCoef6H1', 'IcomplexPolinomialCoef7H1', 'IcomplexPolinomialCoef8H1', 'IcomplexPolinomialCoef9H1', 'IcomplexPolinomialCoef10H1', 'IcomplexPolinomialCoef1H2', 'IcomplexPolinomialCoef2H2', 'IcomplexPolinomialCoef3H2', 'IcomplexPolinomialCoef4H2', 'IcomplexPolinomialCoef5H2', 'IcomplexPolinomialCoef6H2', 'IcomplexPolinomialCoef7H2', 'IcomplexPolinomialCoef8H2', 'IcomplexPolinomialCoef9H2', 'IcomplexPolinomialCoef10H2', 'OentropyH0', 'OentropyH1', 'OentropyH2', 'OlandscapeAmplitudeH0', 'OlandscapeAmplitudeH1', 'OlandscapeAmplitudeH2', 'OnumberOfPointsH0', 'OnumberOfPointsH1', 'OnumberOfPointsH2', 'ObettiAmplitudeH0', 'ObettiAmplitudeH1', 'ObettiAmplitudeH2', 'OcomplexPolinomialCoef1H0', 'OcomplexPolinomialCoef2H0', 'OcomplexPolinomialCoef3H0', 'OcomplexPolinomialCoef4H0', 'OcomplexPolinomialCoef5H0', 'OcomplexPolinomialCoef6H0', 'OcomplexPolinomialCoef7H0', 'OcomplexPolinomialCoef8H0', 'OcomplexPolinomialCoef9H0', 'OcomplexPolinomialCoef10H0', 'OcomplexPolinomialCoef1H1', 'OcomplexPolinomialCoef2H1', 'OcomplexPolinomialCoef3H1', 'OcomplexPolinomialCoef4H1', 'OcomplexPolinomialCoef5H1', 'OcomplexPolinomialCoef6H1', 'OcomplexPolinomialCoef7H1', 'OcomplexPolinomialCoef8H1', 'OcomplexPolinomialCoef9H1', 'OcomplexPolinomialCoef10H1', 'OcomplexPolinomialCoef1H2', 'OcomplexPolinomialCoef2H2', 'OcomplexPolinomialCoef3H2', 'OcomplexPolinomialCoef4H2', 'OcomplexPolinomialCoef5H2', 'OcomplexPolinomialCoef6H2', 'OcomplexPolinomialCoef7H2', 'OcomplexPolinomialCoef8H2', 'OcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef10H2', 'UentropyH0', 'UentropyH1', 'UentropyH2', 'UlandscapeAmplitudeH0', 'UlandscapeAmplitudeH1', 'UlandscapeAmplitudeH2', 'UnumberOfPointsH0', 'UnumberOfPointsH1', 'UnumberOfPointsH2', 'UbettiAmplitudeH0', 'UbettiAmplitudeH1', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef1H0', 'UcomplexPolinomialCoef2H0', 'UcomplexPolinomialCoef3H0', 'UcomplexPolinomialCoef4H0', 'UcomplexPolinomialCoef5H0', 'UcomplexPolinomialCoef6H0', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef8H0', 'UcomplexPolinomialCoef9H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef2H1', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1', 'UcomplexPolinomialCoef6H1', 'UcomplexPolinomialCoef7H1', 'UcomplexPolinomialCoef8H1', 'UcomplexPolinomialCoef9H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef1H2', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.64      0.70      0.67        10
#            1       0.67      0.60      0.63        10

#     accuracy                           0.65        20
#    macro avg       0.65      0.65      0.65        20
# weighted avg       0.65      0.65      0.65        20


# Confusion Matrix:
#  [[7 3]
#  [4 6]]

# ___SFS____:

# Nº of features selected: 15
# Selected features: ['AlandscapeAmplitudeH0', 'ElandscapeAmplitudeH1', 'EbettiAmplitudeH1', 'IcomplexPolinomialCoef10H1', 'OentropyH1', 'OcomplexPolinomialCoef10H2', 'UentropyH0', 'UcomplexPolinomialCoef2H2', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef4H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef6H2', 'UcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef9H2', 'UcomplexPolinomialCoef10H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.75      0.60      0.67        10
#            1       0.67      0.80      0.73        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[6 4]
#  [2 8]]
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___RFECV____:

# Nº of features selected: 2
# Selected features: ['ElandscapeAmplitudeH1', 'UcomplexPolinomialCoef6H1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.80      0.73        10
#            1       0.75      0.60      0.67        10

#     accuracy                           0.70        20
#    macro avg       0.71      0.70      0.70        20
# weighted avg       0.71      0.70      0.70        20


# Confusion Matrix:
#  [[8 2]
#  [4 6]]
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['EcomplexPolinomialCoef10H0', 'IcomplexPolinomialCoef7H1', 'IcomplexPolinomialCoef10H1', 'IcomplexPolinomialCoef1H2', 'IcomplexPolinomialCoef2H2', 'IcomplexPolinomialCoef4H2', 'IcomplexPolinomialCoef6H2', 'IcomplexPolinomialCoef8H2', 'IcomplexPolinomialCoef9H2', 'OcomplexPolinomialCoef6H1', 'OcomplexPolinomialCoef9H1', 'OcomplexPolinomialCoef5H2', 'OcomplexPolinomialCoef7H2', 'UcomplexPolinomialCoef1H1', 'UcomplexPolinomialCoef6H1']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.83      0.50      0.62        10
#            1       0.64      0.90      0.75        10

#     accuracy                           0.70        20
#    macro avg       0.74      0.70      0.69        20
# weighted avg       0.74      0.70      0.69        20


# Confusion Matrix:
#  [[5 5]
#  [1 9]]