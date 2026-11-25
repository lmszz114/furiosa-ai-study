import numpy as np
import pandas as pd

#1. 데이터
import joblib
path = './_save/joblib/'
df = joblib.load(path + 'm22_df_iris.joblib')
print(df)

print("============== 상관관계 히트맵 ==============")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()