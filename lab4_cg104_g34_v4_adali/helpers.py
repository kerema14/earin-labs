import numpy as np
import pandas as pd

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if substring in big_string:
            return substring
    return np.nan
def replace_titles(df):
    title=df['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if df['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
    
def preprocess_data_test(df:pd.DataFrame): #Preprocess test data for kaggle submission
    
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
    df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))
    df['Title']=df.apply(replace_titles, axis=1)
    df['is_wife'] = df['Title'].map(lambda x: 1 if x == 'Mrs' else 0)
    
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Cabin'] = df['Cabin'].astype(str)
    df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    df['Family_Size'] = df['SibSp'] + df['Parch']
    df['Age'].fillna(df['Age'].median(), inplace=True)
    feature_names = ["Pclass","Age","Sex","Family_Size","Deck","Title","is_wife"]
    features = pd.get_dummies(df[feature_names])
    
    data_scaled = features.copy()
    
    
    for column in data_scaled.columns:
        data_scaled[column] = (data_scaled[column] - data_scaled[column].mean()) / data_scaled[column].std(ddof=0)
    

    
    return data_scaled