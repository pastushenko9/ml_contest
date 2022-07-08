import pandas as pd

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from utils.preprocessing import split_data

def grid_searching(df, grid, category_features):
    
    x_train, x_test, y_train, y_test = split_data(df)
    
    model = CatBoostClassifier(
        cat_features=category_features,
        eval_metric = 'AUC',
        random_seed=42)

    grid_search_result = model.grid_search(grid,
                                       X = x_train,
                                       y = y_train,
                                       plot=False)
    
    params = grid_search_result['params']
    
    return params

def train_model(df, params, category_features):
    
    model = CatBoostClassifier(cat_features = category_features,
                              iterations = params['iterations'],
                              depth = params['depth'],
                              learning_rate = params['learning_rate'])
    
    x_train, x_test, y_train, y_test = split_data(df)
    
    model.fit(x_train,
              y_train,
              eval_set=(x_test, y_test))

    model.save_model('models/wrong_monitorings_model')
    
    return model