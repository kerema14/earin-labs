import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from main import preprocess_data
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Base parameters valid for all combinations
base_params = {
    'C': [0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100],
    'fit_intercept': [True, False],
    'class_weight': [None, 'balanced'],
    'warm_start': [True, False],
    'random_state': [42],
    'tol': [1e-4],
}

# Create correctly constrained parameter grids
param_grid = [
    # Option 1: liblinear solver with l1 penalty
    {**base_params, 
     'solver': ['liblinear'],
     'penalty': ['l1'],
     'intercept_scaling': [1, 5, 10]  # intercept_scaling is valid for liblinear
    },
    
    # Option 2: liblinear solver with l2 penalty and dual parameter
    {**base_params,
     'solver': ['liblinear'],
     'penalty': ['l2'],
     'dual': [True, False],
     'intercept_scaling': [1, 5, 10]
    },
    
    # Option 3: saga solver with l1 penalty
    {**base_params,
     'solver': ['saga'],
     'penalty': ['l1']
    },
    
    # Option 4: saga solver with l2 penalty
    {**base_params,
     'solver': ['saga'],
     'penalty': ['l2']
    },
    
    # Option 5: saga solver with elasticnet penalty and l1_ratio
    {**base_params,
     'solver': ['saga'],
     'penalty': ['elasticnet'],
     'l1_ratio': [0.1, 0.25, 0.55, 0.7, 0.9]
    },
    
    # Option 6: newton-cg, lbfgs, sag solvers with l2 penalty
    {**base_params,
     'solver': ['newton-cg', 'lbfgs', 'sag'],
     'penalty': ['l2']
    },
    
   
]

logreg = GridSearchCV(
    LogisticRegression(max_iter=10000), 
    param_grid,
    cv=4, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=2,
    return_train_score=True
)


df = pd.read_csv("titanic/train.csv")
X, y = preprocess_data(df)
logreg.fit(X, y)
results = pd.DataFrame(logreg.cv_results_)
results.to_csv("titanic/logreg_grid_search_results.csv", index=False)