# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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

 # Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50,100,200,500], #50,100,200,500
    #nº of trees in the forest. Generally more is better, but increasing training time.
    'criterion': ['gini', 'entropy'], # gini,entropy
    # 'gini': Uses the Gini impurity as the criterion for splitting nodes.
    # 'entropy': Uses information gain (entropy) as the criterion for splitting nodes.
    # 'gini' is often faster, while 'entropy' can sometimes lead to slightly better trees.

    'max_depth': [None],
    # None: Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    # Integer: The maximum depth of the tree. Limits how deep the tree can grow.
    # Smaller values can help prevent overfitting.

    'min_samples_split': [2,5,10], #2,5,10
    # The minimum number of samples required to split an internal node.
    # Larger values can help prevent overfitting by requiring more samples to make a split.

    'min_samples_leaf': [1,2,4], #1,2,4
    # The minimum number of samples required to be at a leaf node.
    # Larger values can smooth the model, especially in regression.

    'max_features': ['log2', 'log2', None], #sqrt,log2,None
    # 'auto': Equivalent to 'sqrt' in classification, 'sqrt' means max_features=sqrt(n_features).
    # 'sqrt': max_features=sqrt(n_features)
    # 'log2': max_features=log2(n_features)
    # None: max_features=n_features.
    # Controls the number of features to consider when looking for the best split.
    # Smaller values introduce more randomness and can help prevent overfitting.

    # Not very usefull with Random Forest, although it could be used to fine tune the trees
    # 'min_impurity_decrease': [0.0, 0.1, 0.2],
    # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    # Helps to prevent overfitting by pruning nodes that don't significantly improve impurity.
    # Impurity meaning they create child nodes that are more homogeneous than the parent node.

    'class_weight': ['balanced_subsample'], # None, balanced_subsample
    # balanced_subsamples balances the weight from the subsample extracted in the bootstrap

    'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1], 
    #   * **ccp_alpha:** Complexity parameter used for Minimal Cost-Complexity Pruning.
    #   * Larger values result in more pruned trees, which can help prevent overfitting.
    #   * 0.0 means no pruning.

}
# Best Parameters: {'ccp_alpha': 0.0, 'class_weight': 'balanced_subsample', 
# 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
tree_classifier = RandomForestClassifier(random_state=42)
# Use GridSearchCV with the custom scorer
grid_search = GridSearchCV(
    tree_classifier,
    param_grid,
    scoring='accuracy', 
    cv=CV,
    n_jobs=-1,
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
# Best Parameters: {'ccp_alpha': 0.05, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 500}
# Best Score: 0.65
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.89      0.84         9
#            1       0.89      0.80      0.84        10

#     accuracy                           0.84        19
#    macro avg       0.84      0.84      0.84        19
# weighted avg       0.85      0.84      0.84        19


# Confusion Matrix:
#  [[8 1]
#  [2 8]]

# RESAMPLE 0.2
# Best Parameters: {'ccp_alpha': 0.0, 'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Best Score: 0.9349999999999999
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.62      0.89      0.73         9
#            1       0.83      0.50      0.62        10

#     accuracy                           0.68        19
#    macro avg       0.72      0.69      0.68        19
# weighted avg       0.73      0.68      0.67        19


# Confusion Matrix:
#  [[8 1]
#  [5 5]]

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'ccp_alpha': 0.05, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
# Best Score: 0.7125
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.40      0.53        10
#            1       0.60      0.90      0.72        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[4 6]
#  [1 9]]
# RESAMPLE 0.2 STANDARIZED
# Best Parameters: {'ccp_alpha': 0.0, 'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
# Best Score: 0.93
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

# featuresPDRed13EV41kHzStride2
# NO RESAMPLE 0.2
# Best Parameters: {'ccp_alpha': 0.1, 'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# Best Score: 0.725
# Validation Metrics:

# Classification Report:
#                precision    recall  f1-score   support

#            0       0.80      0.40      0.53        10
#            1       0.60      0.90      0.72        10

#     accuracy                           0.65        20
#    macro avg       0.70      0.65      0.63        20
# weighted avg       0.70      0.65      0.63        20


# Confusion Matrix:
#  [[4 6]
#  [1 9]]

# Only U
# NO RESAMPLE 0.2
# Best Parameters: {'ccp_alpha': 0.0, 'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 50}
# Best Score: 0.725
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