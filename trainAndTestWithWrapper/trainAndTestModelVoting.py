from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
import pandas as pd
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier 
import matplotlib
from imblearn.over_sampling import SVMSMOTE
from utils import selectFeaturesSFSAndTrainModel

N_FEATURES_TO_SELECT = 15
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
        x, y, test_size=0., stratify=y
    )

# Resample the training dataset to improve predictions
sm = SVMSMOTE(sampling_strategy={0: 100, 1: 100}, m_neighbors=5, k_neighbors=4)
X_train, y_train = sm.fit_resample(X_train,y_train)

 # Define the parameter grid for GridSearchCV
rf = RandomForestClassifier(ccp_alpha=0.0, class_weight='balanced_subsample', criterion='gini', max_depth=None, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=100)
svc = SVC(C=0.1, class_weight=None, degree=2, gamma=1, kernel='poly', probability=True)
hgb = HistGradientBoostingClassifier(class_weight=None, early_stopping= True, l2_regularization=0.1, learning_rate=0.2, loss='log_loss', max_depth=7, max_iter=300, min_samples_leaf=5, n_iter_no_change=10, validation_fraction=0.1)
# lr = LogisticRegression(C=0.0, class_weight='balanced_subsample', criterion='gini', max_depth= None, max_features='log2', min_samples_split=2, n_estimators=100)
mlp = MLPClassifier(activation='relu', alpha=0.001, batch_size=128, early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive', learning_rate_init=0.01, max_iter=300, solver='adam')
param_grid = {
        'estimators': [
            [('rf', rf), ('svc', svc), ('hgb', hgb), ('mlp', mlp)],
            [('rf', rf), ('svc', svc), ('mlp', mlp)],
            [('rf', rf), ('hgb', hgb), ('mlp', mlp)],
            [('svc', svc), ('hgb', hgb), ('mlp', mlp)],
            [('hgb', hgb), ('mlp', mlp)],
            [('rf', rf),  ('mlp', mlp)],
            [('svc', svc),  ('mlp', mlp)],
            [('rf', rf), ('svc', svc), ('hgb', hgb)],
            [('svc', svc), ('hgb', hgb)],
            [('rf', rf), ('hgb', hgb)],
            [('rf', rf), ('svc', svc)],
            [('rf', rf)],
            [('svc', svc)],
            [('hgb', hgb)],
            [('mlp', mlp)],
        ],
        'voting': ['hard', 'soft']
    }


tree_classifier = VotingClassifier(estimators=[])
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

# Calculate evaluation metrics
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print the metrics
print("Validation Metrics:")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)

# pDRed2.5EV8kHzStride2
# NO RESAMPLE 0.2 STANDARIZED

# RESAMPLE 0.2 STANDARIZED