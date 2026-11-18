import pandas as pd
print(pd.__version__)   # 2.3.3

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '2200'],
    ['LG', '2000', '3000'],
    ['네이버', '100', '1500'],
    ['아모레', '500', '1200']
]
index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성  1000  2000
# 059   현대  1100  2200
# 033   LG  2000  3000
# 045  네이버   100  1500
# 023   아모레   500  1200

# print(df[0])    # raise KeyError(key) from err
# print(df['031'])    # raise KeyError(key) from err
print(df['종목명']) # ★★★ 판다스 열행 ★★★ 컬럼이 기준

############# 아모레 시가 출력 #############
# print(df[3, 1]) # 키 에러
# print(df[3][1]) # 키 에러
# print(df['045']['종목명'])  # 키 에러
print(df['종목명']['045'])  # ★★★ 판다스 열행 ★★★

################################################
# loc : 인덱스 기준으로 행 데이터 추출
# iloc : 행 번호를 기준으로 행 데이터 추출
        # ★★★ int loc로 외울것 ★★★
################################################
print("============= 네이버 뽑기 =============")
# print(df.iloc[3])
# print(df.iloc['045'])   # 에러
# print(df.loc[3])    # 에러
print(df.loc['045'])

print("============= 네이버 종가 =============")
print(df.iloc[3,2]) # 1500
print(df.iloc[3]['종가'])   # 1500
# print(df.iloc[3, '종가'])   # 에러

# print(df.loc['045'][2]) # 에러
# print(df.loc['045', 2]) # 에러
print(df.loc['045']['종가'])
print(df.loc['045', '종가'])
print("=======================================")
print(df.iloc[3].iloc[2])   # 1500
print(df.iloc[3].loc['종가'])   # 1500
print(df.loc['045'].loc['종가']) # 1500
print(df.loc['045'].iloc[2])    # 1500