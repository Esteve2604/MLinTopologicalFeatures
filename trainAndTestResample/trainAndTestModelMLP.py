from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier 
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import trainModelWithSampling

CV = 5
N_JOBS = 2
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")

# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") 
df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
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

 # Define the parameter grid for GridSearchCV
param_grid = {
    'samplingModel__sampling_strategy': [{0: 40, 1: 40}],
    'samplingModel__m_neighbors': [4,5,6],
    'samplingModel__k_neighbors': [3,4,5],
    'trainingModel__hidden_layer_sizes': [
         (25,), (50,), (100,),  # Single hidden layer
        (100, 50), (50, 25), # Two hidden layers
        (100, 50, 25), (150, 100, 50),  # Three hidden layers
        (200, 100, 50, 25) # Four hidden layers
    ],
    'trainingModel__activation': ['relu'],  # Activation functions
    'trainingModel__solver': ['adam'],  # Optimization algorithm
    'trainingModel__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],  # L2 regularization
    'trainingModel__learning_rate_init': [0.0001, 0.01],  # Initial learning rate
    'trainingModel__max_iter': [25, 50, 100, 200],  # Maximum iterations #With 300 selects one hidden layer. With 1000 select 2 hidden layers
    'trainingModel__batch_size': [5,10],  # Minibatch size
    'trainingModel__early_stopping': [True],  # Whether to use early stopping
}

model = MLPClassifier(random_state=42)
sm = SVMSMOTE()
trainModelWithSampling(X_train, y_train, X_test, y_test, preprocessor, sm, model, param_grid, CV, N_JOBS)


# featuresPDRed13EV41kHzStride2
# RESAMPLE 0.2
# Best Parameters: {'samplingModel__k_neighbors': 3, 'samplingModel__m_neighbors': 4, 'samplingModel__sampling_strategy': {0: 40, 1: 40}, 'trainingModel__activation': 'relu', 'trainingModel__alpha': 1, 'trainingModel__batch_size': 10, 'trainingModel__early_stopping': True, 'trainingModel__hidden_layer_sizes': (100,), 'trainingModel__learning_rate_init': 0.01, 'trainingModel__max_iter': 25, 'trainingModel__solver': 'adam'}
# Best Score: 0.7375
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