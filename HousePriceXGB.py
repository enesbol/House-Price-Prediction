#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
import joblib
from sklearn.pipeline import Pipeline


# In[ ]:





# In[81]:


pd.options.display.max_rows = 100
pd.options.display.max_columns = None


# In[82]:


df = pd.read_csv("house_price_1.csv")


# In[83]:


df['MSSubClass'] = df['MSSubClass'].astype('object')

dfc =df.copy()

drop_cols = ['MiscFeature','PoolQC']
fill_miss_cols = ['MasVnrArea','Fence','GarageCond','GarageQual','GarageFinish','GarageYrBlt','GarageType','FireplaceQu','Electrical','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','MasVnrType','Alley','LotFrontage']


# In[84]:


for i in fill_miss_cols:
    if dfc[i].dtypes in ["int64","float64"] :
        dfc[i].fillna(-999.0,inplace=True)
    elif dfc[i].dtypes in ["object"] :
        dfc[i].fillna('unknown',inplace=True)


# In[85]:


dfc = dfc.drop(drop_cols,axis=1)


# In[86]:


qual_cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond']

a={'Ex':5,'Gd': 4, 'TA':3,'Fa' : 2, 'Po':1,'NA':0}

dfc[qual_cols]=dfc[qual_cols].replace(a)


# In[87]:


dfc['BsmtExposure'] = dfc['BsmtExposure'].replace({'Gd':4,'Av':3,'Mn':2,'No':1,'NA':0})

dfc['CentralAir'] = dfc['CentralAir'].replace({'N':0,'Y':1})


# In[88]:


dfq = dfc.copy()


# In[89]:


dfq = dfq.drop(columns=['Id','GarageYrBlt','TotRmsAbvGrd','1stFlrSF','ScreenPorch','PoolArea'])


# In[90]:


for i in dfq :
    if (dfq[i].dtypes in ["int64","float64"]) and (abs(dfq.corr()['SalePrice'][i]) < 0.03):
        dfq = dfq.drop(columns = i)
        print(f"{i} kolonu atıldı")


# In[91]:


categorical_columns = dfq.columns[dfq.dtypes==object].tolist()


# In[92]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()


# In[93]:


single_row = dfq.mode(axis=0)
single_row = single_row.iloc[:,:-1]
single_row.to_csv('single_row.csv')


# In[94]:


plus = dfq[:25]
plus = plus.iloc[:,:-1]
plus.to_csv('plus.csv')


# In[95]:


hot = ohe.fit_transform(dfq[categorical_columns].astype(str))


# In[96]:


col_names = list(dfq[categorical_columns].columns)


# In[97]:


dfq[dfq['MSSubClass']==120]


# In[98]:


import joblib
joblib.dump(ohe, 'ohe.joblib')


# In[99]:


hot


# In[100]:


1460*254


# In[101]:


cold_df = dfq.select_dtypes(exclude=["object"])
cold_df.head()


# In[102]:


from scipy.sparse import csr_matrix
cold = csr_matrix(cold_df)


# In[103]:


from scipy.sparse import hstack
final_sparse_matrix = hstack((hot, cold))


# In[104]:


final_df = pd.DataFrame(final_sparse_matrix.toarray())
final_df.head()


# In[105]:


from sklearn.base import TransformerMixin 
class NullValueImputer(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        for column in X.columns.tolist():
            if column in X.columns[X.dtypes==object].tolist():
                X[column] = X[column].fillna(X[column].mode())
            else:
                X[column]=X[column].fillna(-999.0)
        return X


# In[106]:


class SparseMatrix(TransformerMixin):
    def __init__(self):
        None
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        categorical_columns = X.columns[X.dtypes==object].tolist()
        ohe = OneHotEncoder()
        hot = ohe.fit_transform(X[categorical_columns])
        cold_df = X.select_dtypes(exclude=["object"])
        cold = csr_matrix(cold_df)
        final_sparse_matrix = hstack((hot, cold))
        final_csr_matrix = final_sparse_matrix.tocsr()
        return final_csr_matrix


# In[107]:


from sklearn.pipeline import Pipeline
data_pipeline = Pipeline([('null_imputer', NullValueImputer()), ('sparse', SparseMatrix())])


# In[108]:


X_train_transformed = data_pipeline.fit_transform(X_train)


# In[109]:


final_csr_matrix = final_sparse_matrix.tocsr()


# In[110]:


from sklearn.model_selection import train_test_split
X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=2)


# In[111]:


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor


# In[112]:


kfold = KFold(n_splits=5, shuffle=True, random_state=2)


# In[113]:


def cross_val(model):
    scores = cross_val_score(model, 
                             X_train_transformed, 
                             y_train, 
                             scoring='neg_root_mean_squared_error', 
                             cv=kfold)
    rmse = (-scores.mean())
    return rmse


# In[114]:


X_train_transformed = X_train
cross_val(XGBRegressor(missing=-999.0))


# In[115]:


X_train_transformed.head()


# In[116]:


X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train_transformed, y_train, random_state=2)


# In[117]:


def n_estimators(model):
    eval_set = [(X_test_2, y_test_2)]
    eval_metric="rmse"
    model.fit(X_train_2, y_train_2, 
              eval_metric=eval_metric, 
              eval_set=eval_set, 
              early_stopping_rounds=100)
    y_pred = model.predict(X_test_2)
    rmse = MSE(y_test_2, y_pred)**0.5
    return rmse


# In[118]:


n_estimators(XGBRegressor(n_estimators=5000, missing=-999.0))


# Using our default model, 7 estimators currently gives the best estimate. That will be our starting point.
# 
# [7]	validation_0-rmse:42541.29014

# In[119]:


def grid_search(params, reg=XGBRegressor(missing=-999.0)):
    grid_reg = GridSearchCV(reg, params, scoring='neg_mean_squared_error', cv=kfold)
    grid_reg.fit(X_train_transformed, y_train)
    best_params = grid_reg.best_params_
    print("Best params:", best_params)
    best_score = np.sqrt(-grid_reg.best_score_)
    print("Best score:", best_score)


# In[ ]:





# In[120]:


grid_search(params={'max_depth':[1, 2, 3, 4, 6, 7, 8],
                     'n_estimators':[7]})


# In[154]:


grid_search(params={'max_depth':[7, 8, 9,10],
                    'min_child_weight':[1,2,3,4,5],
                    'n_estimators':[7]})


# In[158]:


grid_search(params={'max_depth':[9],
                    'min_child_weight':[4,5],
                    'subsample':[0.5, 0.6, 0.7, 0.8, 0.9],
                    'n_estimators':[7, 50]})


# In[159]:


grid_search(params={'max_depth':[8],
                    'min_child_weight':[3, 4],
                    'subsample':[0.6, 0.7, 0.8],
                    'colsample_bytree':[0.6, 0.7, 0.8, 0.9],
                    'n_estimators':[50]})


# In[160]:


grid_search(params={'max_depth':[8],
                    'min_child_weight':[4],
                    'subsample':[.8],
                    'colsample_bytree':[0.8],
                    'colsample_bylevel':[0.6, 0.7, 0.8, 0.9, 1],
                    'colsample_bynode':[0.6, 0.7, 0.8, 0.9, 1],
                    'n_estimators':[50]})


# In[164]:


grid_search(params={'max_depth':[7],
                    'min_child_weight':[4],
                    'subsample':[.8],
                    'colsample_bytree':[0.8],
                    'colsample_bylevel':[0.8],
                    'colsample_bynode':[0.6],
                    'n_estimators':[50]})


# In[121]:


xgbr = XGBRegressor(max_depth=7, 
             min_child_weight=4, 
             subsample=0.8, 
             colsample_bytree=0.8, 
             colsample_bylevel=0.8, 
             colsample_bynode=0.6,
             n_estimators =100,
             missing=-999.0,
            learning_rate=0.1)


# In[122]:


xgbr.fit(X_train.values,y_train.values)


# In[123]:


pred = xgbr.predict(X_test.values)


# In[124]:


from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score

mse = MSE(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))
print(f"R2: {r2}")


# In[125]:


fe = xgbr.feature_importances_


# In[126]:


len(fe)


# In[127]:


ilk20 =a.head(20).index


# In[128]:


ilk20 = list(ilk20)


# In[129]:


ilk20=['ExterQual',
 'OverallQual',
 'GarageCars',
 'GrLivArea',
 'BsmtQual',
 'FireplaceQu',
 'FullBath',
 'KitchenQual',
 'CentralAir',
 'TotalBsmtSF',
 'BsmtFinSF1',
 'Alley',
 'LandContour',
 '2ndFlrSF',
 'KitchenAbvGr',
 'GarageArea',
 'Condition1',
 'Neighborhood',
 'GarageQual',
 'PavedDrive']


# In[130]:


ilk20 = list(map(lambda x: x.replace('x33_N', 'PavedDrive'), ilk20))


# In[131]:


plt.barhdf(df_names[sorted_idx], xgb.featureimportances[sorted_idx])
plt.xlabel("Xgboost Feature Importance")


# In[132]:


col_names[33]


# In[ ]:





# In[133]:


ilk20


# In[134]:


a = pd.DataFrame(xgbr.feature_importances_,
                                   index = tot,
                                    columns=['importance']).sort_values('importance',ascending=False)
a.head(20)


# In[135]:


names = ohe.get_feature_names() 


# In[136]:


len(names)


# In[137]:


xgbr.save_model('housepricexgb.model')


# In[138]:


len(cold_df.columns)


# In[139]:


tot = list(names)+list(cold_df.columns)


# In[140]:


tot


# In[141]:


fea_list = list(feature_importances.index)


# In[142]:


for i in fea_list:
    print({tot[i]})


# In[143]:


tot.pop()


# In[144]:


len(tot)


# In[145]:


feature_importance.head()


# In[146]:


print('sa')


# In[147]:


a


# In[148]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
for i in ilk20 :
    print(f"{i} tipi {dfq[i].dtypes}")
    if dfq[i].dtypes in ["int64","float64"] :
        print(f"{i} standart sapması : {np.std(dfq[i])}")        
        plt.scatter(dfq[i],dfq["SalePrice"])        
        plt.show()
        #df.iplot(x=i, y='SalePrice',kind="scatter",mode='markers', size=8,colorscale="red")
        dfq[[i]].iplot(kind="hist")
    elif dfq[i].dtypes in ["object"] :
        print(f"{i} değişkenindeki unique değer sayısı : {dfq[i].squeeze().nunique()} adettir.")
        df2 = dfq.groupby(i).agg({"SalePrice" : ["mean",np.median,"count"]})
        df2.columns = ["mean","median","count"]
        #plt.scatter(df[i],df["SalePrice"])
        #plt.show()
        df2[["mean","median"]].iplot(kind="box")
        df2[["count"]].iplot(kind="bar")


# In[149]:


from sklearn.metrics import r2_score


# In[150]:


y_pred = xgbr.predict(X_test.values)


# In[151]:


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))
#MAE: 26745.1109986


# In[156]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("")
plt.legend()
plt.show()


# In[180]:


plt.xlabel('TotalBsmtSF',size=15.5)
plt.ylabel('SalePrice',size=15.5)
plt.title("TotalBsmtSF-SalePrice",size=17)
plt.scatter(dfq["TotalBsmtSF" ],dfq["SalePrice"])        
plt.show()


# In[ ]:




