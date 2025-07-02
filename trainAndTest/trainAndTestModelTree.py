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
N_JOBS = -1

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
    'criterion': ['gini', 'entropy'],
    # 'gini': Uses the Gini impurity as the criterion for splitting nodes.
    # 'entropy': Uses information gain (entropy) as the criterion for splitting nodes.
    # 'gini' is often faster, while 'entropy' can sometimes lead to slightly better trees.

    'splitter': ['best', 'random'],
    # 'best': Chooses the best split at each node.
    # 'random': Chooses the best random split at each node.
    # 'random' can be useful for reducing overfitting and increasing speed, especially for high-dimensional data.

    'max_depth': [None, 5, 10, 15, 20],
    # None: Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    # Integer: The maximum depth of the tree. Limits how deep the tree can grow.
    # Smaller values can help prevent overfitting.

    'min_samples_split': [2, 5, 10],
    # The minimum number of samples required to split an internal node.
    # Larger values can help prevent overfitting by requiring more samples to make a split.

    'min_samples_leaf': [1, 2, 4],
    # The minimum number of samples required to be at a leaf node.
    # Larger values can smooth the model, especially in regression.

    'max_features': ['sqrt', 'log2', None],
    # 'auto': Equivalent to 'sqrt' in classification, 'sqrt' means max_features=sqrt(n_features).
    # 'sqrt': max_features=sqrt(n_features)
    # 'log2': max_features=log2(n_features)
    # None: max_features=n_features.
    # Controls the number of features to consider when looking for the best split.
    # Smaller values introduce more randomness and can help prevent overfitting.

    'max_leaf_nodes': [None, 10, 20, 30],
    # None: Unlimited number of leaf nodes.
    # Integer: The maximum number of leaf nodes.
    # Limits the growth of the tree.

    'min_impurity_decrease': [0.0, 0.1, 0.2],
    # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    # Helps to prevent overfitting by pruning nodes that don't significantly improve impurity.
    # Impurity meaning they create child nodes that are more homogeneous than the parent node.

    'class_weight': [None],
}

tree_classifier = DecisionTreeClassifier(random_state=42)

# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    tree_classifier,
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
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': 5, 'max_features': 'log2', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'random'}
# Best Score: 0.6910714285714284
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.62      0.56      0.59         9
#            1       0.64      0.70      0.67        10

#     accuracy                           0.63        19
#    macro avg       0.63      0.63      0.63        19
# weighted avg       0.63      0.63      0.63        19


# Confusion Matrix:
#  [[5 4]
#  [3 7]]
# RESAMPLE 0.2
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'random'}
# Best Score: 0.9
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.56      0.56      0.56         9
#            1       0.60      0.60      0.60        10

#     accuracy                           0.58        19
#    macro avg       0.58      0.58      0.58        19
# weighted avg       0.58      0.58      0.58        19


# Confusion Matrix:
#  [[5 4]
#  [4 6]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'class_weight': None, 'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 2, 'splitter': 'best'}
# Best Score: 0.825
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.56      0.50      0.53        10
#            1       0.55      0.60      0.57        10

#     accuracy                           0.55        20
#    macro avg       0.55      0.55      0.55        20
# weighted avg       0.55      0.55      0.55        20


# Confusion Matrix:
#  [[5 5]
#  [4 6]]
# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 30, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'random'}
# Best Score: 0.8850000000000001
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

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'max_leaf_nodes': 10, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'splitter': 'random'}
# Best Score: 0.7875
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

# ONLY U
# NO RESAMPLE 0.2
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': 5, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 5, 'splitter': 'best'}
# Best Score: 0.75
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

# Filtrado Univariante
# Best Parameters: {'class_weight': None, 'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt', 'max_leaf_nodes': 20, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
# Best Score: 0.7875
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