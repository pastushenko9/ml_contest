import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

def clean_data(df):
    df.fillna(0)
    df.drop_duplicates(inplace=True)
    df = df.query('day_of_week == 1')
    df = df.sort_values(by='date')
    df.drop(['day_of_week', 'date'], axis = 1, inplace=True)
    
    return df

def add_features(df):

    df['price_deviation'] = abs(1 - (df['price'] / df['day_mean_price']))
    df['product_percetage_checkmons'] = df['product_count_checkmons'] / df['product_count_mons']
    df['shop_percetage_checkmons'] = df['shop_count_checkmons'] / df['shop_count_mons']
    df['competitor_percetage_checkmons'] = df['competitor_count_checkmons'] / df['competitor_count_mons']
    
    return df

def split_data(df):
    x_train, x_test, y_train, y_test = train_test_split(df.drop('wrong_monitoring', axis = 1),
                                                        df['wrong_monitoring'],
                                                        test_size=0.1,
                                                        random_state = 55) 

    return x_train, x_test, y_train, y_test

def feature_importances_analysis(df):
    
    x_train, x_test, y_train, y_test = split_data(df)
    
    model = CatBoostClassifier(
        cat_features=['shop_id','product_name', 'task_name',\
                  'competitor', 'region_id', 'city_id', 'group_id', \
                  'category_id', 'task_comment', 'product_comment'], 
        iterations = 250,
        random_seed=42
    )

    model.fit(x_train,
              y_train,
              eval_set=(x_test, y_test)
    )
    
    feature_names = x_train.columns
    
    return model, feature_names

def importance_feature(model, feature_names):
    
    score_data = []
    feature_importances = model.feature_importances_
    for score, name in sorted(zip(feature_importances, feature_names), reverse=False):
        score_data.append([round(score, 2), name])
    score_data.append([1.0, 'wrong_monitoring'])

    return score_data

def processed_df(df, score_data):
    features = pd.DataFrame(data=score_data, columns=['score', 'feature'])
    features = features.query('score > 0.05')
    df = df[features['feature'].unique()]   

    return df

def actual_cat_features(df):
    all_cat_features = [
        'shop_id','product_name', 'task_name', \
        'competitor', 'region_id', 'city_id', 'group_id', \
        'category_id', 'task_comment', 'product_comment'] 
    
    category_features = []
    for column in df.columns:
        if column in all_cat_features:
            category_features.append(column)
            
    return category_features