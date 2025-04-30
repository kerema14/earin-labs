#Kerem Adalı, only and only.
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helpers import preprocess_data_test, substrings_in_string,replace_titles
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
import os


SEED = 69
overfitting_threshold = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def preprocess_data(df:pd.DataFrame):
    df["target"] = df["Survived"]
    df["target"] = df["target"].astype(int)
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df.apply(replace_titles, axis=1) #replace niche titles with more common ones
    df['is_wife'] = df['Title'].map(lambda x: 1 if x == 'Mrs' else 0) #add a column to indicate if the passenger is a wife
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown'] #list of cabin letters
    df['Cabin'] = df['Cabin'].astype(str)
    df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Age'].fillna(df['Age'].median(), inplace=True)
    feature_names = ["Pclass","Age","Sex","Family_Size","Deck","Title","is_wife"]
    features = pd.get_dummies(df[feature_names]) #get dummies for categorical variables
    data_scaled = features.copy()
    
    for column in data_scaled.columns:
        data_scaled[column] = (data_scaled[column] - data_scaled[column].mean()) / data_scaled[column].std(ddof=0) #normalize the data
    

    
    return data_scaled,df["target"]




def predict_kaggle():
    df = pd.read_csv("titanic/train.csv")
    test_csv = pd.read_csv("titanic/test.csv")
    X, y = preprocess_data(df)
    Z = preprocess_data_test(test_csv)
    Z = Z.reindex(columns=X.columns, fill_value=False)
    logreg = LogisticRegression(random_state=SEED, C=0.25, class_weight=None, fit_intercept=True, intercept_scaling=1, penalty='l1', solver='liblinear', tol=0.0001, warm_start=True) 
    #{'C': 0.25, 'class_weight': None, 'fit_intercept': True, 'intercept_scaling': 1, 'penalty': 'l1', 'random_state': 42, 'solver': 'liblinear', 'tol': 0.0001, 'warm_start': True}

    final_logreg = logreg.fit(X, y)
    predictions1 = final_logreg.predict(Z)
    print(predictions1)
    final_prediction_csv = pd.DataFrame({
        "PassengerId": 892+np.arange(len(predictions1)),
        "Survived": predictions1
    })
    final_prediction_csv.to_csv("titanic/submission.csv", index=False)



    
   
def main(seed):
    overfitting_count = 0
    # DONE Load and preprocess dataset
    df = pd.read_csv("titanic/train.csv")
    X, y = preprocess_data(df)
    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # DONE Define the models
    logreg = LogisticRegression(random_state=seed, C=0.25, class_weight=None, fit_intercept=True, intercept_scaling=1, penalty='l1', solver='liblinear', tol=0.0001, warm_start=True) 
    #best params: {'C': 0.25, 'class_weight': None, 'fit_intercept': True, 'intercept_scaling': 1, 'penalty': 'l1', 'random_state': 42, 'solver': 'liblinear', 'tol': 0.0001, 'warm_start': True}
    
    dtree = DecisionTreeClassifier(random_state=seed,ccp_alpha=0.0, class_weight=None, criterion='gini', max_depth=20, max_features=None, min_impurity_decrease=0.0, min_samples_leaf=2, min_samples_split=20, splitter='random')
    #best params: {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 20, 'random_state': 42, 'splitter': 'random'}


    
    # DONE evaluate model using cross-validation
    scores1 = cross_val_score(logreg, X_train, y_train, cv=4, scoring='f1')
    scores2 = cross_val_score(dtree, X_train, y_train, cv=4, scoring='f1')

    # Fit the best model on the entire training set and get the predictions
    final_logreg = logreg.fit(X_train, y_train)
    final_dtree = dtree.fit(X_train, y_train)

    predictions1 = final_logreg.predict(X_test)
    predictions2 = final_dtree.predict(X_test)
    
    print("Cross Validation Logistic Regression F1 Score: ", scores1.mean())
    print("Cross Validation Decision Tree F1 Score: ", scores2.mean())


    # DONE Evaluate the final predictions with the metric of your choice

    # Evaluate Logistic Regression predictions
    f1 = f1_score(y_test, predictions1)
    accuracy = accuracy_score(y_test, predictions1)
    precision = precision_score(y_test, predictions1)
    recall = recall_score(y_test, predictions1)

    print("\nEvaluation Metrics for Logistic Regression:")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    dtf1 = f1_score(y_test, predictions2)
    dtaccuracy = accuracy_score(y_test, predictions2)
    dtprecision = precision_score(y_test, predictions2)
    dtrecall = recall_score(y_test, predictions2)

    print("\nEvaluation Metrics for Decision Tree:")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    if (abs(f1-scores1.mean()) > overfitting_threshold):
        print("Logistic Regression is overfitting")
        overfitting_count += 1
        
        
        
    
    if (abs(dtf1-scores2.mean()) > overfitting_threshold):
        print("Decision Tree is overfitting")
        overfitting_count += 1
        
        
        
    
    return [f1,accuracy,precision,recall], [dtf1,dtaccuracy,dtprecision,dtrecall], overfitting_count

if __name__ == "__main__":
    # Set seed for reproducibility
    """
    print("Creating predictions for Kaggle submission...")
    predict_kaggle()
    print("Predictions created. Press Enter to continue...")
    input()
    """
    
    logreg_f1 = []
    dtree_f1 = []
    logreg_accuracy = []
    dtree_accuracy = []
    logreg_precision = []
    dtree_precision = []
    logreg_recall = []
    dtree_recall = []
    overfitting_count = 0
    random_seeds = [
        31, 69, 420, 1337, 9001, 42, 123, 666, 777, 888,
        1010, 1234, 4321, 8675, 309, 314, 2718, 1984, 2001, 2024,
        9999, 1001, 8080, 5555, 4444, 3333, 2222, 1111, 707, 909,
        1024, 2048, 4096, 8192, 137, 911, 1600, 1776, 1492, 1999,
        73, 27, 54, 108, 216, 69, 84, 96, 360, 720,
        144, 69, 420, 5, 10, 15, 20, 25, 30, 35,
        40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
        90, 95, 100, 111, 121, 131, 141, 151, 161, 171,
        181, 191, 201, 211, 221, 231, 241, 251, 261, 271,
        281, 291, 301, 313, 323, 333, 343, 353, 363, 373
    ] #100 different seeds
    random_seeds = [i * SEED for i in random_seeds]

    for random_seed in random_seeds: #accumulate results accross 100 different seed iterations
        set_seed(random_seed)
        lgres,dtres,ofc = main(seed=random_seed)
        overfitting_count += ofc
            
        logreg_f1.append(lgres[0])
        dtree_f1.append(dtres[0])
        logreg_accuracy.append(lgres[1])
        dtree_accuracy.append(dtres[1])
        logreg_precision.append(lgres[2])
        dtree_precision.append(dtres[2])
        logreg_recall.append(lgres[3])
        dtree_recall.append(dtres[3])
  
    
    print("\nLogistic Regression F1 Scores mean: ", np.mean(logreg_f1))
    print("Logistic Regression Accuracies mean: ", np.mean(logreg_accuracy))
    print("Logistic Regression Recalls mean: ", np.mean(logreg_recall))
    print("Logistic Regression Precisions mean: ", np.mean(logreg_precision))
    print("\nDecision Tree F1 Scores mean: ", np.mean(dtree_f1))
    print("Decision Tree Accuracies mean: ", np.mean(dtree_accuracy))
    print("Decision Tree Precisions mean: ", np.mean(dtree_precision))
    print("Decision Tree Recalls mean: ", np.mean(dtree_recall))
    print("Overfitting Count: "+str(overfitting_count)+" out of 200 fits")
    # Prepare DataFrames for the metrics
    df_f1 = pd.DataFrame({
        "Logistic Regression": logreg_f1,
        "Decision Tree": dtree_f1
    })
    df_accuracy = pd.DataFrame({
        "Logistic Regression": logreg_accuracy,
        "Decision Tree": dtree_accuracy
    })
    df_precision = pd.DataFrame({
        "Logistic Regression": logreg_precision,
        "Decision Tree": dtree_precision
    })
    df_recall = pd.DataFrame({
        "Logistic Regression": logreg_recall,
        "Decision Tree": dtree_recall
    })

    # Create figures directory if it doesn't exist
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Plot boxplot for F1 Scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_f1)
    plt.title("F1 Scores Box Plot Across Iterations")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "f1_scores_boxplot.png"))
    plt.show()

    # Plot boxplot for Accuracies
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_accuracy)
    plt.title("Accuracies Box Plot Across Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "accuracies_boxplot.png"))
    plt.show()

    # Plot boxplot for Precisions
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_precision)
    plt.title("Precisions Box Plot Across Iterations")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "precisions_boxplot.png"))
    plt.show()

    # Plot boxplot for Recalls
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_recall)
    plt.title("Recalls Box Plot Across Iterations")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "recalls_boxplot.png"))
    plt.show()

