import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x,columns=datasets.feature_names)
df['Target'] = y
# print(df)

#joblib, pickle로 save
path = './_save/joblib/'
import joblib
joblib.dump(df,path+'m22_df_iris.joblib')
path = './_save/pickle/'
import pickle
pickle.dump(df,open(path+'m22_df_iris.pickle','wb'))