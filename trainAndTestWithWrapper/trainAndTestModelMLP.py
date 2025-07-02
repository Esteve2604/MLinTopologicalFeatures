from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier 
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndTrainModel

N_JOBS = 2
N_FEATURES_TO_SELECT = 15
CV = 2
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

datasetPath = "./data/afeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/efeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ifeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ofeaturesPDRed13EV41kHzStride2.csv"
# datasetPath = "./data/ufeaturesPDRed13EV41kHzStride2.csv"

df = pd.read_csv(datasetPath)
print("El dataset utilizado es: " + datasetPath)

# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
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


model = MLPClassifier(random_state=42, activation='relu', alpha=0.0001, batch_size=10, early_stopping=True, hidden_layer_sizes=(100,50), learning_rate_init=0.01, max_iter=25, solver='adam')

selectFeaturesSFSAndTrainModel(X_train, y_train,X_test, y_test, preprocessor, model, N_FEATURES_TO_SELECT, CV, N_JOBS)


# ___SFS____:

# Nº of features selected: 15
# Selected features: ['ElandscapeAmplitudeH0', 'EbettiAmplitudeH2', 'EcomplexPolinomialCoef3H0', 'EcomplexPolinomialCoef7H0', 'EcomplexPolinomialCoef2H2', 'EcomplexPolinomialCoef4H2', 'EcomplexPolinomialCoef8H2', 'IcomplexPolinomialCoef3H0', 'IcomplexPolinomialCoef9H0', 'ObettiAmplitudeH1', 'OcomplexPolinomialCoef6H0', 'OcomplexPolinomialCoef8H2', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef4H1', 'UcomplexPolinomialCoef5H1']
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
# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# ___SFS____:

# Nº of features selected: 15
# Selected features: ['EcomplexPolinomialCoef6H2', 'IentropyH0', 'OcomplexPolinomialCoef9H0', 'OcomplexPolinomialCoef9H2', 'UentropyH1', 'UbettiAmplitudeH2', 'UcomplexPolinomialCoef5H0', 'UcomplexPolinomialCoef7H0', 'UcomplexPolinomialCoef9H0', 'UcomplexPolinomialCoef10H0', 'UcomplexPolinomialCoef3H1', 'UcomplexPolinomialCoef10H1', 'UcomplexPolinomialCoef3H2', 'UcomplexPolinomialCoef5H2', 'UcomplexPolinomialCoef7H2']
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.80      0.80        10
#            1       0.80      0.80      0.80        10

#     accuracy                           0.80        20
#    macro avg       0.80      0.80      0.80        20
# weighted avg       0.80      0.80      0.80        20


# Confusion Matrix:
#  [[8 2]
#  [2 8]]