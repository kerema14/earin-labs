import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from main import preprocess_data

# Create parameter grid for DecisionTreeClassifier
dt_param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 4, 6, 8, 10, 15, 20],
    'min_samples_split': [2, 4, 8, 12, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': [None, 'sqrt', 'log2'],
    'random_state': [42],
    'class_weight': [None, 'balanced'],
    'min_impurity_decrease': [0.0, 0.01, 0.05],
    'ccp_alpha': [0.0, 0.01, 0.05]
}

# Create GridSearchCV object
dt_grid = GridSearchCV(
    DecisionTreeClassifier(), 
    dt_param_grid,
    cv=4, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=2,
    return_train_score=True
)

# Load and preprocess data
df = pd.read_csv("titanic/train.csv")
X, y = preprocess_data(df)

# Fit the model
dt_grid.fit(X, y)

# Save results
results = pd.DataFrame(dt_grid.cv_results_)
results.to_csv("titanic/dt_grid_search_results.csv", index=False)