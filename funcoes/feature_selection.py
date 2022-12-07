
import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
#import eli5
#from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

## Pearson:
def Correlacao(X, y, num_feats):
    '''
    Seleciona features por Correlação
    '''
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]

    return cor_support, cor_feature

## VarianceThreshold
def Variancia(X, y, num_feats):
    '''
    Seleciona features por Limite de variância
    '''

    embeded_var_selector = VarianceThreshold(threshold=(.1))
    embeded_var_selector.fit(X)

    embeded_var_support = embeded_var_selector.get_support()
    embeded_var_feature = X.loc[:,embeded_var_support].columns.tolist()
    #print(str(len(embeded_rf_feature)), 'selected features')

    
    return embeded_var_support, embeded_var_feature
## Lasso:
def Lasso(X, y, num_feats):
    '''
    Seleciona features por penalização L1 em regressão Logistica
    '''
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), max_features=num_feats)
    embeded_lr_selector.fit(X, y)

    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    #print(str(len(embeded_lr_feature)), 'selected features')
    
    return embeded_lr_support, embeded_lr_feature

## Floresta:
def Floresta(X, y, num_feats):
    '''
    Seleciona features usando random forest
    '''
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    #print(str(len(embeded_rf_feature)), 'selected features')
    
    return embeded_rf_support, embeded_rf_feature

def Ensamble(X, y, num_feats):
    '''
    Seleciona features usando Ensamble LightGBM
    '''
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgb_selector.fit(X, y)

    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    return embeded_lgb_support, embeded_lgb_feature

## RFE:
def Recursivo(X, y, num_feats, model):
    '''
    Seleciona features usando estimador fornecido
    '''

    cor_list = []
    feature_name = X.columns.tolist()

    rfe_selector = RFE(estimator=model, n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    #print(str(len(rfe_feature)), 'selected features') 

    return rfe_support, rfe_feature


from sklearn.inspection import permutation_importance

## Permutation importance
def Permutacao(X, y, num_feats, model):
    '''
    Seleciona features usando permutação
    '''
    model.fit(X, y)
    r = permutation_importance(model, X, y,
                           n_repeats=5,
                           random_state=42)
    embeded_permutation_support = np.where(r.importances_mean>0, True, False)
    embeded_permutation_feature = X.loc[:,embeded_permutation_support].columns.tolist()
    #print(str(len(embeded_rf_feature)), 'selected features')

    return embeded_permutation_support, embeded_permutation_feature

def Feature_selection(X, y, num_feats, model=None):
    '''

    '''
    feature_name = X.columns.tolist()
    X_norm = MinMaxScaler().fit_transform(X)
    X_norm = pd.DataFrame(X_norm, columns=feature_name)
    ## Pearson
    cor_support, cor_feature = Correlacao(X, y, num_feats)
    ## VarianceThreshold
    embeded_var_support, embeded_var_feature = Variancia(X, y, num_feats)
    ## Lasso
    embeded_lr_support, embeded_lr_feature = Lasso(X_norm, y, num_feats)
    ## Floresta
    embeded_rf_support, embeded_rf_feature = Floresta(X, y, num_feats)
    ## Ensamble
    embeded_lgb_support, embeded_lgb_feature = Ensamble(X, y, num_feats)
    ## RFE
    rfe_support, rfe_feature = Recursivo(X_norm, y, num_feats, model)
    ## Permutation
    embeded_permutation_support, embeded_permutation_feature = Permutacao(X_norm, y, num_feats, model)


    # put all selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 
                                         'Pearson':cor_support, 
                                         'Variancia':embeded_var_support, 
                                         'Lasso':embeded_lr_support,
                                         'Random Forest':embeded_rf_support,
                                         'Ensamble':embeded_lgb_support,
                                         'RFE':rfe_support, 
                                         'Permutação':embeded_permutation_support
                                         })

    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    #feature_selection_df.head(num_feats)
    return feature_selection_df
