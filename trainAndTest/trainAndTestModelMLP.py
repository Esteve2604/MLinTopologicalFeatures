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

N_JOBS = -1
CV = 5
# df = pd.read_csv("./data/featuresPDRed2.5EV8kHzStride2.csv")
# df = pd.read_csv("./data/featuresPDRed13EV41kHzStride2.csv")
# df = pd.read_csv("./data/selected_ThresholdfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdAfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdEfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdIfeaturesPDRed13EV44,1kHzStride2.csv")
# df = pd.read_csv("./data/univariate/selected_ThresholdOfeaturesPDRed13EV44,1kHzStride2.csv")
df = pd.read_csv("./data/univariate/selected_ThresholdUfeaturesPDRed13EV44,1kHzStride2.csv")

# df = pd.read_csv("./data/afeaturesPDRed13EV41kHzStride2.csv") 
# df = pd.read_csv("./data/efeaturesPDRed13EV41kHzStride2.csv") 
# df = pd.read_csv("./data/ifeaturesPDRed13EV41kHzStride2.csv") 
# df = pd.read_csv("./data/ofeaturesPDRed13EV41kHzStride2.csv") 
# df = pd.read_csv("./data/ufeaturesPDRed13EV41kHzStride2.csv") 
# df.drop("sampleName", axis=1, inplace=True)

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
        'hidden_layer_sizes': [
         (25,), (50,), (100,),  # Single hidden layer
        (100, 50), (50, 25), # Two hidden layers
        (100, 50, 25), (150, 100, 50),  # Three hidden layers
        (200, 100, 50, 25) # Four hidden layers
    ],
    'activation': ['relu'],  # Activation functions
    'solver': ['adam'],  # Optimization algorithm
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],  # L2 regularization
    'learning_rate_init': [0.0001, 0.01],  # Initial learning rate
    'max_iter': [25, 50, 100, 200],  # Maximum iterations #With 300 selects one hidden layer. With 1000 select 2 hidden layers
    'batch_size': [5,10],  # Minibatch size
    'early_stopping': [True],  # Whether to use early stopping
}

mlp_classifier = MLPClassifier(random_state=42)

# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    mlp_classifier,
    param_grid,
    scoring='accuracy', 
    cv=CV,
    n_jobs=N_JOBS,
    verbose=1
)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                        ('gridSearch', grid_search)])
pipeline.fit(X_train, y_train)

#Extracting best score and parameters
best_params = pipeline.named_steps['gridSearch'].best_params_
best_score = pipeline.named_steps['gridSearch'].best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
best_tree = pipeline.named_steps['gridSearch'].best_estimator_

#Training the best model obtained
X_test = preprocessor.transform(X_test)
y_pred = best_tree.predict(X_test)
# decisionTree = tree.DecisionTreeClassifier(class_weight={0:2,1:1})
# decisionTree.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = decisionTree.predict(X_test)

# Calculate evaluation metrics
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the metrics
print("Validation Metrics:")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# NO RESAMPLE 0.2
# Best Parameters: {'activation': 'relu', 'alpha': 0, 'batch_size': 10, 'early_stopping': True, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.6767857142857142
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.67      0.67         9
#            1       0.70      0.70      0.70        10

#     accuracy                           0.68        19
#    macro avg       0.68      0.68      0.68        19
# weighted avg       0.68      0.68      0.68        19


# Confusion Matrix:
#  [[6 3]
#  [3 7]]

# RESAMPLE 0.2
# Best Parameters: {'activation': 'relu', 'alpha': 0.001, 'batch_size': 10, 'early_stopping': True, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.915
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.71      0.56      0.62         9
#            1       0.67      0.80      0.73        10

#     accuracy                           0.68        19
#    macro avg       0.69      0.68      0.68        19
# weighted avg       0.69      0.68      0.68        19


# Confusion Matrix:
#  [[5 4]
#  [2 8]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 10, 'early_stopping': True, 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.7571428571428571
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.53      0.53      0.53        15
#            1       0.53      0.53      0.53        15

#     accuracy                           0.53        30
#    macro avg       0.53      0.53      0.53        30
# weighted avg       0.53      0.53      0.53        30


# Confusion Matrix:
#  [[8 7]
#  [7 8]]
# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 5, 'early_stopping': True, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.9049999999999999
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.53      0.67      0.59        15
#            1       0.55      0.40      0.46        15

#     accuracy                           0.53        30
#    macro avg       0.54      0.53      0.52        30
# weighted avg       0.54      0.53      0.52        30


# Confusion Matrix:
#  [[10  5]
#  [ 9  6]]

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'activation': 'relu', 'alpha': 0.01, 'batch_size': 10, 'early_stopping': True, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.725
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

# ONLY U
# NO RESAMPLE
# Best Parameters: {'activation': 'relu', 'alpha': 0.1, 'batch_size': 10, 'early_stopping': True, 'hidden_layer_sizes': (150, 100, 50), 'learning_rate_init': 0.01, 'max_iter': 25, 'solver': 'adam'}
# Best Score: 0.725
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.67      0.60      0.63        10
#            1       0.64      0.70      0.67        10

#     accuracy                           0.65        20
#    macro avg       0.65      0.65      0.65        20
# weighted avg       0.65      0.65      0.65        20


# Confusion Matrix:
#  [[6 4]
#  [3 7]]