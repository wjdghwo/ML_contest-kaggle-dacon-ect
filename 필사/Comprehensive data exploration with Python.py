# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:36:57 2020

@author: Administrator
"""

'''
Competition Description


주택 구입자에게 꿈의 집을 설명해달라고 요청하면, 지하 천장 높이나 동서 철도와의 근접성으로 시작하지 않을 것입니다.
그러나 이 대회의 데이터 세트는 침실 수나 흰 울타리보다 가격 협상에 훨씬 더 많은 영향을 미치는 어떤것을 증명합니다.

아이오와 주 에임스에 있는 주거용 주택의 거의 모든 측면을 설명하는 79개의 설명 변수가있는 이 경쟁에서는 각 주택의 최종 가격을 예측해야 합니다.

Practice Skills

Creative feature engineering 
Advanced regression techniques like random forest and gradient boosting
'''

"""
SalePrice : 부동산 판매가격. 예측하려는 목표 변수입니다.
MSSubClass: 건물 등급
MSZoning : 일반 구역 분류
LotFrontage : 사유지에 연결된 도로의 직선 피트
LotArea : 평방 피트 단위의 부지 크기
Street : 도로 접근 유형
Alley : 골목 접근 유형
LotShape : 사유지의 일반적인 모양
LandContour : 사유지의 평탄도
Utilities : 사용 가능한 유틸리티 유형
LotConfig : 품목 구성
LandSlope : 사유지의 경사
Neighborhood : 에임스시 경계 내의 물리적 위치
Condition1 : 주요 도로 또는 철도와의 근접성
Condition2 : 주요 도로 또는 철도와의 근접성 (짧은거리일 경우)
BldgType : 주거 유형
HouseStyle : 주거 스타일
OverallQual : 전체 재료 및 마감 품질
OverallCond : 전체 상태 등급
YearBuilt : 원래 건축 날짜
YearRemodAdd : 리모델링 날짜
RoofStyle : 지붕 유형
RoofMatl : 지붕 재료
Exterior1st : 집 외부 자제
Exterior2nd : 집 외부 자제 (하나 이상의 재료인 경우)
MasVnrType : 벽돌 베니어 유형
MasVnrArea : 벽돌 베니어 면적 (평방 피트)
ExterQual : 외장재 품질
ExterCond : 외장재의 현황
Foundation : 토대 유형
BsmtQual : 지하 높이
BsmtCond : 지하실의 일반 상태
BsmtExposure : 워크 아웃 또는 정원 수준의 지하 벽
BsmtFinType1 : 지하실 마감면의 품질
BsmtFinSF1 : 유형 1 마감 평방 피트
BsmtFinType2 : 두 번째 완성 된 영역의 품질 (있는 경우)
BsmtFinSF2 : 유형 2 마감 평방 피트
BsmtUnfSF : 미완성 된 지하실 면적
TotalBsmtSF : 지하 총 평방 피트
Heating : 난방 유형
HeatingQC : 난방 품질 및 상태
CentralAir : 중앙 에어컨
Electrical : 전기 시스템
1stFlrSF : 1 층 평방 피트
2ndFlrSF : 2 층 평방 피트
LowQualFinSF : 저품질 마감 평방 피트 (모든 층)
GrLivArea : 지상 (지상) 거실 면적 평방 피트
BsmtFullBath : 지하 전체 욕실
BsmtHalfBath : 지하 반 욕실
FullBath : 상급 화장실
HalfBath : 상급 화장실(세면대와 변기만 있는)
Bedroom : 지하층 이상의 침실 수
Kitchen : 주방 수
KitchenQual : 주방 품질
TotRmsAbvGrd : 상급 전체 방 (화장실 제외)
Functional : 홈 기능 등급
Fireplaces : 벽난로 수
FireplaceQu : 벽난로 품질
GarageType : 차고 위치
GarageYrBlt : 차고 건설 연도
GarageFinish : 차고 내부 마감
GarageCars : 차량 수용 가능 차고 크기
GarageArea : 차고 크기 (평방 피트)
GarageQual : 차고 품질
GarageCond : 차고 상태
PavedDrive : 포장 된 진입로
WoodDeckSF : 목재 데크 면적 (평방 피트)
OpenPorchSF : 평방 피트 단위의 열린 현관 영역
EnclosedPorch : 닫힌 현관 영역 (평방 피트)
3SsnPorch : 평방 피트 단위의 3 계절 현관 면적
ScreenPorch : 스크린 현관 영역 (평방 피트)
PoolArea : 수영장 면적 (평방 피트)
PoolQC : 수영장 품질
Fence : 울타리 품질
MiscFeature : 다른 카테고리에서 다루지 않는 기타 기능
MiscVal : 기타 기능의 달러 가치
MoSold : 월 판매
YrSold : 판매 연도
SaleType : 판매 유형
SaleCondition : 판매 조건
"""

# In[ ]
'''
'인생에서 가장 어려운 것은 자신을 아는 것이다.'

이 인용문은 탈레스의 말입니다.
탈레스는 과학적 사상에 종사하고 이를 즐긴 것으로 알려진 서양 문명의 최초의 인물로 인정받은
그리스 / 페니키아 철학자, 수학자 및 천문학자입니다.
(출처 : https://en.wikipedia.org/wiki/Thales).

데이터를 아는 것이 데이터 과학에서 가장 어려운 일이라고 말하지는 않지만 많은 시간이 걸리는 작업입니다.
따라서 이 초기 단계를 간과하고 너무 빨리 물에 뛰어 들기 쉽습니다.

그래서 나는 물에 뛰어 들기 전에 수영하는 법을 배우려고 노력했습니다.
Hair et al.(2013)의 'Examining your data' 챕터에서 데이터 분석을 완벽하진 않지만 최선을 다해 이해하고자 했습니다.
이 커널에 대한 철저한 연구를 보고하는 것은 아니지만 커뮤니티에 유용 할 수 있기를 바랍니다.
그래서 저는 이 문제에 데이터 분석 원칙을 어떻게 적용했는지를 공유하고 있습니다.

챕터로 입력한 이상한 이름에도 불구하고 이 커널에서 우리가 하는 일은 다음과 같습니다.

1.문제 이해.
우리는 각 변수를 살펴보고 이 문제에 대한 의미와 중요성에 대한 철학적 분석을 수행할 것입니다.

2.일변량 검토.
종속 변수 ('SalePrice')에 초점을 맞추고 그것에 대해 조금 더 알아 보려고합니다.

3.다변량 검토.
우리는 종속 변수와 독립 변수가 어떻게 관련되는지 이해하려고 노력할 것입니다.

4.기초 정리.
데이터 세트를 정리하고 결손 데이터, 이상치 및 범주 형 변수를 처리합니다.

5.가정 검토
데이터가 대부분의 다변량 기법에서 요구하는 가정을 충족하는지 확인합니다.

이제 재미있게 놀 시간입니다!
'''

# In[ ]
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# In[ ]
#bring in the six packs
df_train = pd.read_csv('./house/train.csv')

# In[ ]
#check the decoration
print(df_train.columns)

# In[ ]
'''
1. 그래서... 우리는 무엇을 기대할 수 있습니까?
데이터를 이해하기 위해, 각 변수를 살펴볼 수 있고 그 의미와 문제와의 관련성을 이해하려고 노력할 수 있습니다.
시간이 많이 소요된다는 것을 알고 있지만, 그럿은 데이터 세트의 느낌을 줄 것입니다.

분석에 몇 가지 원칙을 적용하기 위해, 다음 열이있는 Excel 스프레드 시트를 만들 수 있습니다.

Variable - 변수 이름.

Type - 변수 유형 확인.
이 필드에는 두 가지 값이 있을 수 있습니다 : 'numerical' or 'categorical'.
'numerical'는 값이 숫자인 변수를 의미하고 'categorical'는 값이 범주인 변수를 의미합니다.

Segment - 변수의 세그먼트(부분) 식별.
우리는 세 개의 가능한 세그먼트를 정의할 수 있습니다 : 건물, 공간 또는 위치 
'건물'이란, 건물의 물리적 특성과 관련된 변수를 의미합니다 (예 : 'OverallQual').
'공간'이란, 집의 공간 속성을 알리는 변수를 의미합니다 (예 : 'TotalBsmtSF').
마지막으로 '위치'란, 집이 있는 장소에 대한 정보를 제공하는 변수를 의미합니다 (예 : 'Neighborhood').

Expectation - 'SalePrice'의 변수 영향에 대한 우리의 기대.
가능한 값인 '높음', '중간' 및 '낮음'으로 범주형 척도를 사용할 수 있습니다.

Conclusion -데이터를 간략히 살펴본 후 변수의 중요성에 대한 결론.
우리는 'Expectation'에서 동일한 범주형 척도로 유지할 수 있습니다.

Comments -우리에게 발견된 일반적인 의견입니다.

'Type'과 'Segment'는 향후 참고 용일 뿐이지만 'Expectation'열은 'sixth sense'를 개발하는 데 도움이 되기 때문에 중요합니다.
이 열을 채우려면, 모든 변수에 대한 설명을 읽고 하나씩 고민해봐야합니다.

집을 살 때 이 변수에 대해 생각합니까?
(예 : 우리가 꿈꾸는 집에 대해 생각할 때 'Masonry veneer type'에 관심이 있습니까?)

그렇다면 이 변수가 얼마나 중요할까요?
(예 : 'Poor'대신 'Excellent'소재가 외관에 미치는 영향은 무엇입니까?
'Good'대신 'Excellent'를 사용하면 어떤 영향을 미칩니까?)

이 정보가 이미 다른 변수에 설명되어 있습니까? 
(예 : 'LandContour'가 건물의 평탄도를 제공하는 경우 실제로 'LandSlope'를 알아야합니까?).

이 벅찬 연습을 마치면 스프레드 시트를 필터링하고 'High' 'Expectation'로 변수를 주의깊게 살펴볼 수 있습니다.
그런 다음 해당 변수와 'SalePrice'사이의 산점도에 들어가서 기대치의 수정본을 'Conclusion'열에 채울 수 있습니다.

이 과정을 거쳐 다음 변수가 이 문제에서 중요한 역할을 할 수 있다는 결론을 내렸습니다.

OverallQual
(계산 방법을 모르기 때문에 싫어하는 변수입니다.
재미있는 연습은 사용 가능한 다른 모든 변수를 사용하여 'OverallQual'을 예측하는 것입니다).
YearBuilt.
TotalBsmtSF.
GrLivArea.

결국 두 개의 'building' 변수('OverallQual'과 'YearBuilt')와 두 개의 'space' 변수('TotalBsmtSF'와 'GrLivArea')를 갖게 되었다. 
이 것은 중요한건 'location, location and location'가 전부이다라는 실제 부동산 주문에 위배되기 때문에 약간 예상치 못한 일입니다.
이 빠른 데이터 검사 프로세스는 범주형 변수에 대해 약간 가혹했을 수 있습니다.
예를 들어, 'Neigborhood' 변수가 관련성이 더 높을 것으로 예상했지만 데이터 검사 후 결국 제외되었습니다.
아마 이것은 범주형 변수 시각화에 더 적합한 상자 그림 대신에 산점도의 사용에 관련이 있을 수 있습니다.
데이터를 시각화하는 방식은 종종 결론에 영향을 미칩니다.

하지만 이 연습의 요점은 데이터와 기대치에 대해 조금 생각하는 것이기 때문에 목표를 달성했다고 생각합니다.
이제 '조금 덜 대화하고, 조금 더 행동 해주세요'를 할 때입니다.
서두릅시다!
'''

# In[ ]
'''
2. 가장 먼저 해야 할 일: 'SalePrice' 분석

'SalePrice'는 우리 탐구의 이유입니다.
우리가 파티에 갈 때와 같습니다.
우리는 항상 거기에 있을 이유가 있습니다.
일반적으로 여성이 그 이유입니다.
(공지사항 : 선호도에 따라 남성, 춤 또는 알코올로 맞게 조정해라)

여성을 비유로 들어 'SalePrice를 만난 방법'에 대한 작은 이야기를 만들어 보겠습니다.

댄스 파트너를 찾고 있을 때 Kaggle 파티에서 모든 것이 시작되었습니다.
잠시동안 댄스 플로어에서 둘러보니 우리는 바 근처에서 댄스 신발을 신고있는 소녀를 보았습니다.
그것은 그녀가 춤을 추기 위해 거기에 있다는 신호입니다.
우리는 예측 모델링을하고 분석 대회에 참여하는 데 많은 시간을 소비하므로 소녀들과 이야기하는 것은 우리의 초능력이 아닙니다.
그럼에도 불구하고 우리는 그것을 시도했습니다:

'안녕, 난 Kaggly야! 넌? 'SalePrice'? 얼마나 아름다운 이름입니까! 
당신은 'SalePrice'를 알고 있지, 너에 대한 데이터를 좀 줄 수 있니?
두 사람의 성공적인 관계 가능성을 계산하는 모델을 개발했어.
우리에게 적용하고 싶어!'
'''

# In[ ]
#descriptive statistics summary
print(df_train['SalePrice'].describe())

# In[ ]
'''
'좋아요 ... 최저가는 0보다 큰 것 같습니다.
훌륭해! 너는 내 모델을 파괴할 개인적인 특성을 가지고 있지 않습니다!
저에게 보낼 수 있는 사진이 있습니까?
몰라요 ... 해변에 계신가요 ... 아니면 체육관에서 셀카를 찍으셨나요?'
'''

# In[ ]
#histogram
sns.distplot(df_train['SalePrice']);

# In[ ]
'''
'아! 외출 할 때 seaborn 화장을 하시는 것 같네요 ... 
너무 우아해요! 나는 또한 보인다 :

정규 분포에서 벗어납니다.
상당한 양의 왜도가 있습니다.
뾰족함을 보여줍니다.

흥미로워지고 있습니다!
'SalePrice', 몸매 측정해 주시겠어요? '
'''

# In[ ]
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew()) # 왜도 측정
print("Kurtosis: %f" % df_train['SalePrice'].kurt()) # 첨도 측정

# In[ ]
'''
'Amazing! 내 사랑 계산기가 맞다면 우리의 성공 확률은 97.834657%야.
우리 다시 만나야 할 것 같아!
다음 주 금요일에 시간 있으면 내 번호로 연락 줘.
이따 봐, crocodile!'
'''

# In[ ]
'''
'SalePrice', her buddies and her interests

싸울 지형을 선택하는 것이 군사적 지혜입니다.
'SalePrice'가 떠나자마자 우리는 페이스북에 갔습니다.
그래, 이제 심각해지고 있어.
이것은 스토킹이 아닙니다.
내 말뜻을 알고 있다면, 그것은 단지 한 사람의 열정적인 조사일 뿐이다.

그녀의 프로필에 따르면, 우리는 몇 명의 공통적인 친구가 있다.
척 노리스 외에 우리 둘 다 'GrlivArea'와 'TotalBsmtSF'를 알고 있다.
더구나 우리는 'OverallQual'과 'YearBuilt'와 같은 공통 관심사를 가지고 있다.
이것은 조짐이 좋아보인다!

우리의 연구를 최대한 활용하기 위해, 우리는 공통의 친구들의 프로필을 주의 깊게 살펴보는 것으로 시작할 것이고, 나중에는 공통의 관심사에 초점을 맞출 것이다.
'''

# In[ ]
'''
Relationship with numerical variables
'''
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1) # 열 방향으로 데이터 합치기
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000)); # y측은 0부터 800000까지
# 파이썬에서 세미콜론은 여러 구문을 이어쓸때 세미콜론을 쓴다. 대부분 언어들이 구문이 끝나면 ;을 붙이지만 파이썬은 붙일 필요 없다. 붙여도 오류가 나지도 않는다.
'''
음... 'SalePrice'와 'GrLivArea'는 정말 오랜 친구로 선형적인 관계를 보여.

'TotalBsmtSF'는 어떻니?
'''

# In[ ]
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# In[ ]
'''
'TotalBsmtSF'는 또한 'SalePrice'의 절친이지만 좀 더 감성적인 관계처럼 보입니다!
모든 것이 정상이고 갑자기 강한 선형(exponential?)반응으로 모든 것이 바뀝니다.
게다가, 때때로 'TotalBsmtSF'가 자기자신에게 꼭 맞고 'SalePrice'에 0점을 준다.
'''
# In[ ]
'''
Relationship with categorical features
'''
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
'''
모든 예쁜 여자들처럼, 'SalePrice'는 'OverallQual'를 즐깁니다.
Note to self: McDonald가 첫 데이트로 적절한지 생각해봐.
'''

# In[ ]
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90); # x축의 틱(tick)편집/x축 변수이름들 90도로 회전

'''
강한 경향은 아니지만 'SalePrice'는 오래된 유물보다 새로운 것에 더 많은 돈을 쓰는 경향이 있다고 말하고 있습니다.

Note : 'SalePrice'가 일정한 가격인지 여부는 알 수 없습니다.
일정한 가격은 인플레이션의 영향을 제거하려고 합니다.
'SalePrice'가 일정한 가격이 아니라면 보다 더 가격이 수년에 걸쳐 비교할 수 있도록 해야 합니다.
'''

# In[ ]
'''
In summary

이야기를 제외하고 우리는 다음과 같은 결론을 내릴 수 있습니다.

'GrLivArea'와 'TotalBsmtSF'는 'SalePrice'와 선형적으로 관련되어있는 것 같습니다.
두 관계 모두 양수이므로 한 변수가 증가하면 다른 변수도 증가합니다.
'TotalBsmtSF'의 경우, 선형 관계의 기울기가 특히 높다는 것을 알 수 있습니다.

'OverallQual'과 'YearBuilt'도 'SalePrice'와 관련이 있는 것 같습니다.
상자 그림이 전체 품질에 따라 판매 가격이 어떻게 상승하는지 보여주는 'OverallQual'의 경우 관계가 더 강해 보입니다.

방금 4개의 변수를 분석했지만 분석해야 할 다른 변수가 많이 있습니다.
여기서 트릭은 올바른 피처들(피처 선택)을 선택하는 것이지 피처 간의 복잡한 관계 정의(피처 엔지니어링)하는 것은 아닌 것 같습니다.

즉, 겉껍질에서 밀을 분리합시다.
'''

# In[ ]
'''
3. 침착하고 현명하게 일하라

지금까지 우리는 직관에 따라 중요하다고 생각하는 변수를 분석했습니다.
분석에 객관적인 성격을 부여하려는 노력에도 불구하고, 우리의 출발점은 주관적이라고 말해야합니다.

엔지니어로서 저는 이 접근 방식이 마음에 들지 않습니다.
모든 나의 교육은 주관성의 바람을 견딜 수 있는 훈련된 마음을 발달하는 것이었습니다.
그 이유가 있습니다.
구조 공학에서 주관적으로 해보면 만든 것이 부족한 물리학을 볼 수 있습니다.
다칠 수 있습니다.

그래서 관성을 극복하고 더 객관적인 분석을 해보자.
'''

# In[ ]
'''
The 'plasma soup'

'처음에는 plasma soup 외에는 아무것도 없었습니다.
우주론 연구가 시작될 때의 이 짧은 순간에 대해 알려진 것은 대부분 추측입니다.
그러나 과학은 오늘날 우주에 대해 알려진 것을 바탕으로 아마도 일어난 일에 대한 스케치를 고안했습니다.'
(출처 : http://umich.edu/~gs265/bigbang.htm)

우주를 탐험하기 위해 우리는 'plasma soup'를 이해하기 위한 몇 가지 실용적인 방안으로 시작할 것입니다 :

상관 행렬 (heatmap style).
'SalePrice'상관 행렬 (zoomed heatmap style).
상관 관계가 가장 높은 변수 사이의 산점도 (move like Jagger style).
'''

# In[ ]
'''
Correlation matrix (heatmap style)

'''
#correlation matrix
corrmat = df_train.corr() # 상관관계를 corrmat에 저장
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True); 
# vmax=.8 : 컬러맵 0.8에 고정 / square=True : heatmap 정사각형으로 크기조정

'''
제 생각에, 이 히트 맵은 'plasma soup'와 그 관계에 대한 간략한 개요를 얻을 수 있는 가장 좋은 방법입니다.(감사합니다 @seaborn!)

첫 눈에 내 관심을 끄는 두 개의 빨간색 사각형이 있습니다.
첫 번째는 'TotalBsmtSF'와 '1stFlrSF'변수를 참조하고 두 번째는 'GarageX'변수를 참조합니다.
두 경우 모두 이러한 변수 간의 상관 관계가 얼마나 중요한지 보여줍니다.
실제로 이 상관 관계는 너무 강해서 다중공선성의 상황을 나타낼 수 있습니다.
이러한 변수에 대해 생각하면, 거의 동일한 정보를 제공하므로 다중공선성이 실제로 발생한다는 결론을 내릴 수 있습니다.
히트 맵은 이러한 종류의 상황을 감지하는데 유용하며, 우리와 같이 피처 선택이 가장 중요한 특징이 되는 문제에서 필수적인 도구입니다.

제가 주목 한 또 다른 점은 'SalePrice' 상관 관계였습니다.
잘 알려진 'GrLivArea', 'TotalBsmtSF', 'OverallQual'이 크게 '안녕!'이라 말하는 것을 볼 수 있지만 고려해야 할 다른 많은 변수도 볼 수 있습니다.
이것이 우리가 다음에 할 일입니다.
'''


# In[ ]
'''
'SalePrice' correlation matrix (zoomed heatmap style)
'''

#saleprice correlation matrix
k = 10 # number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # 상관관계 top10 추출
cm = np.corrcoef(df_train[cols].values.T) # 피어슨 상관계수 추출
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#cbar : 컬러 바(오른쪽 막대)를 그릴 지 여부
# annot : True이면 각 셀에 데이터 값을 씁니다
# fmt='.2f' : 주석 추가에 필요한 문자열 형식 코드/소숫점 둘째자리까지 표기
# annot_kws={'size': 10} : annot=True일 때 ax.text에 대한 키워드 인수입니다/글자 크기 조정
# yticklabels=cols.values, xticklabels=cols.values # x, y tick에 변수명 지정
plt.show()

# In[ ]
'''
수정 구슬에 따르면 이것이 'SalePrice'와 가장 관련이 있는 변수입니다.
이것에 대한 나의 생각 :

'OverallQual', 'GrLivArea'및 'TotalBsmtSF'는 'SalePrice'와 강한 상관 관계가 있습니다. 체크하세요!
'GarageCars'와 'GarageArea'도 가장 강력한 상관 변수 중 일부입니다.
그러나 마지막 하위 요점에서 논의했듯이 차고에 수용가능한 차량 수는 차고 크기의 결과입니다.
'GarageCars'와 'GarageArea'는 쌍둥이 형제와 같습니다.
당신은 그들을 구별 할 수 없을 것입니다.
따라서 분석에서 이러한 변수 중 하나만 필요합니다.
('SalePrice'와의 상관 관계가 더 높기 때문에 'GarageCars'를 유지할 수 있음).
'TotalBsmtSF'와 '1stFloor'도 쌍둥이 형제 인 것 같습니다.
우리는 우리의 첫 번째 추측이 옳았다고 말하기 위해 'TotalBsmtSF'를 유지할 수 있습니다.
('So ... What can we expect?'를 다시 읽어주세요).
'FullBath'?? 정말?
'TotRmsAbvGrd'와 'GrLivArea', 다시 쌍둥이 형제.
이 데이터 셋은 체르노빌의 데이터입니까?
아 ... 'YearBuilt'... 'YearBuilt'는 'SalePrice'와 약간의 상관 관계가있는 것 같습니다.
솔직히 'YearBuilt'에 대해 생각하는 것이 무섭습니다.
이 문제를 해결하기 위해 약간의 시계열 분석을해야한다는 느낌이 들기 때문입니다.
나는 이것을 숙제로 남겨 둘 것이다.

산점도를 살펴 보겠습니다.
'''

# In[ ]
'''
'SalePrice'와 상관변수 사이의 산점도 (move like Jagger style)

당신이 보게 될 것을 준비하십시오.
이 산점도를 처음 보았을 때 나는 완전히 날아갔다고 고백해야합니다!
아주 짧은 공간에 많은 정보가 있습니다. 정말 놀랍습니다.
다시 한번 @seaborn에게 감사드립니다!
당신은 나를 'move like Jagger'로 만듭니다!
'''

# In[ ]
#scatterplot
sns.set() # 그래프를 그릴 set 생성
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5) # pairplot : 각 변수(variable)별로 각각의 상관관계를 표현, 대각성분은 변수의 히스토그램, 나며지는 산점도
plt.show();

# In[ ]
'''
우리는 이미 몇 가지 주요 수치를 알고 있지만, 이 메가 산점도는 변수 관계에 대한 합리적인 아이디어를 제공합니다.

흥미로운 수치 중 하나는 'TotalBsmtSF'와 'GrLiveArea'사이의 수치입니다.
이 그림에서 우리는 거의 테두리처럼 작동하는 선형 선을 그리는 점들을 볼 수 있습니다.
대부분의 점이 그 선 아래에 머무르는 것은 완전히 의미가 있습니다.
지하실은 지상 생활 공간과 동일 할 수 있지만, 지상 생활 공간보다 더 큰 지하 공간은 예상되지 않습니다. 
(벙커를 구입하려는 경우 제외).

'SalePrice'와 'YearBuilt'에 관한 줄거리도 우리를 생각하게 만들 수 있습니다.
'dots cloud'의 맨 아래에서 우리는 거의 부끄러운 지수 함수 (창의적이 되십시오)로 보이는 것을 볼 수 있습니다.
우리는 또한 'dots cloud'의 상한선에서도 동일한 경향을 볼 수 있습니다 (더욱 창의적이 되십시오).
또한 지난 몇 년 동안의 점 세트가이 한도를 초과하는 경향이있는 경향이 있습니다.
(지금 가격이 더 빠르게 상승하고 있다고 말하고 싶었습니다).

좋아, 지금은 Rorschach 테스트가 충분하다.
결측치 : 결측치로 이동해 보겠습니다!
'''

# In[ ]
'''
4. Missing data

결측치에 대해 생각할 때 중요한 질문 :

결측치가 얼마나 많이 발생합니까?
결측치가 무작위입니까 아니면 패턴이 있습니까?

결측치는 표본 크기의 감소를 의미할 수 있기 때문에 이러한 질문에 대한 대답은 실용적인 이유로 중요합니다.
이로 인해 분석을 진행하지 못할 수 있습니다.
또한 실질적인 관점에서 결측치 프로세스가 편향되지 않고 불편한 진실을 숨기지 않도록 해야합니다.
'''

# In[ ]
#missing data
total = df_train.isnull().sum().sort_values(ascending=False) # 변수별 결측치 개수
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False) # 변수별 결측치 비율
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # 결측치 개수와 비율 데이터를 합침
print(missing_data.head(20))

# In[ ]
'''
결측치를 처리하는 방법을 이해하기 위해 이를 분석해봅시다.

데이터의 15% 이상이 누락된 경우 해당 변수를 삭제하고 존재하지 않는 것처럼 해야한다고 생각합니다.
이는 이러한 경우 결측치를 채우기 위해 어떤 트릭도 시도하지 않을 것임을 의미합니다.
이에 따라 삭제해야 할 변수 집합 (예 : 'PoolQC', 'MiscFeature', 'Alley'등)이 있습니다.
요점은 이 데이터를 누락시킬 것인가? 나는 그렇게 생각하지 않습니다.
이러한 변수는 대부분 우리가 집을 살 때 생각하는 측면이 아니기 때문에 매우 중요하지 않은 것 같습니다.(아마도 데이터가 누락된 이유일까요?).
또한 변수를 자세히 살펴보면 'PoolQC', 'MiscFeature'및 'FireplaceQu'와 같은 변수가 이상치에 대한 강력한 후보라고 말할 수 있으므로 기꺼이 삭제할 수 있습니다.

나머지 경우와 관련하여 'GarageX'변수에 동일한 수의 결측치가 있음을 알 수 있습니다.
결측치는 관측치들의 동일한 세트와 관련이 있다고 확신합니다.
(확인하지는 않겠지만, 5%에 ​​불과하며 25% 문제를 사용해서는 안됩니다).
차고에 관한 가장 중요한 정보는 'GarageCars'로 표현되며, 결측치의 5%에 ​​불과하다는 점을 고려하여 언급된 'GarageX'변수를 삭제하겠습니다.
동일한 논리가 'BsmtX'변수에 적용됩니다.

'MasVnrArea'및 'MasVnrType'에 대해서는 이러한 변수가 필수가 아니라는 점을 고려할 수 있습니다.
또한 이미 고려중인 'YearBuilt'와 'OverallQual'변수와 강한 상관 관계가 있습니다.
따라서 'MasVnrArea'및 'MasVnrType'을 삭제해도 정보가 손실되지 않습니다.

마지막으로 'Electrical'에 결측치가 하나 있습니다.
하나의 관측치이므로 이 관측치를 삭제하고 변수를 유지합니다.

요약하면 결측치를 처리하기 위해 'Electrical'변수를 제외하고 결측치가 있는 모든 변수를 삭제합니다.
'Electrical'에서는 결측치가 있는 관측치를 삭제합니다.
'''

# In[ ]
#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1) # 결측치가 1개 이상인 변수들 제거
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) # 결측치가 1개 있는 변수는 데이터 값 제거
print(df_train.isnull().sum().max()) # just checking that there's no missing data missing...

# In[ ]
'''
이상치
이상치는 또한 우리가 알아야 할 사항입니다. 왜?
이상치는 모델에 현저한 영향을 미칠 수 있고 중요한 정보 소스가 되어 특정 행동에 대한 통찰력을 제공할 수 있기 때문입니다.

이상치는 복잡한 주제이며 더 많은 주의를 기울여야합니다.
여기서는 'SalePrice'의 표준편차와 산점도 세트를 통해 빠른 분석을 수행합니다.
'''

# In[ ]
'''
일변량 분석
여기서 주요 관심사는 관측치를 이상치로 정의하는 임계값을 설정하는 것입니다.
이를 위해 데이터를 표준화할 것입니다.
이 맥락에서 데이터 표준화는 데이터 값을 평균이 0이고 표준편차가 1이되도록 변환하는 것을 의미합니다.
'''

# In[ ]
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
# StandardScaler(). # 표준화
# fit_transform( # 표준화모델을 fit()한 후, 본 데이터를 transform()으로 바꾸어 주는데 이 과정을 한 번에 합침
# df_train['SalePrice'][:,np.newaxis]); # SalePrice 변수를 표준화함/ np.newaxis으로 1459->(1459,1)으로 차원을 증가시킴

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10] # 낮은 그룹의 이상치 집단 / argsort : 작은값부터 순서대로 데이터의 위치를 변환
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:] # 높은 그룹의 이상치 집단
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# In[ ]

'''
'SalePrice'가 새 옷으로 어떻게 보이는지 :

낮은 범위 값은 비슷하며 0에서 그리 멀지 않습니다.
높은 범위 값은 0에서 멀리 떨어져 있고 7.xxx 값은 실제로 범위를 벗어납니다.
지금은 이러한 값을 이상치로 간주하지 않지만이 두 7.xxx 값에 주의해야합니다.
'''

# In[ ]
'''
이변량 분석

우리는 이미 다음과 같은 산점도를 알고 있습니다.
그러나 우리가 새로운 관점에서 사물을 바라 볼 때 항상 발견할 것이 있습니다.
Alan Kay가 말했듯이 '관점의 변화는 IQ 80 포인트의 가치가 있습니다'.
'''

# In[ ]
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# In[ ]
'''
공개된 내용 :

더 큰 'GrLivArea'와 두 값이 이상해 보이며 집단을 따르지 않습니다.
왜 이런 일이 일어나는지 추측할 수 있습니다.
아마도 그것들은 시골을 얘기하고 그 곳이 낮은 가격임을 말해줍니다.
나는 이것에 대해 확실하지 않지만이 두 가지 점이 일반적인 경우를 대표하지 않는다고 확신합니다.
따라서 우리는 그것들을 이상치로 정의하고 삭제할 것입니다.

플롯의 맨 위에있는 두 가지 관찰값은 우리가 주의해야 한다고 말한 7.xxx의 관찰값입니다.
두 가지 특별한 경우처럼 보이지만 추세를 따르는 것 같습니다.
그런 이유로 우리는 그것들을 남길 것입니다.
'''

# In[ ]
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2] # GrLivArea기준으로 내림차순 정렬해서 상위 2개 추출
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index) # 상위 2개의 데이터 제거
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

# In[ ]
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# In[ ]
'''
일부 관찰값(예 : TotalBsmtSF> 3000)을 제거하고 싶은 유혹을 느낄 수 있지만 그만한 가치가 없다고 생각합니다.
우리는 그것으로 살 수 있으므로 아무것도하지 않을 것입니다.
'''

# In[ ]
'''
5. 하드 코어 얻기

Ayn Rand의 소설 'Atlas Shrugged'에는 자주 반복되는 질문이 있습니다.
John Galt는 누구입니까?
이 책의 큰 부분은 이 질문에 대한 답을 찾기 위한 탐구에 관한 것입니다.

나는 지금 Randian을 느낀다. 'SalePrice'는 누구입니까?

이 질문에 대한 답은 다변량 분석을 위한 통계적 근거의 기초가 되는 가정에 대한 테스트에 있습니다.
우리는 이미 데이터 정리를 했고 'SalePrice'에 대해 많은 것을 발견했습니다.
이제 'SalePrice'가 다변량 기법을 적용할 수 있는 통계적 가정을 어떻게 준수하는지 깊이 이해해야 할 때입니다.

Hair et al.(2013)에 따르면, 네 가지 가정을 테스트해야합니다.

정규성 - 정규성에 대해 이야기 할 때 의미하는 것은 데이터가 정규분포처럼 보여야한다는 것입니다.
    여러 통계 테스트가 이에 의존하기 때문에 중요합니다. (예 : t- 통계량)
    이 연습에서는 'SalePrice'(제한된 접근 방식)에 대한 일변량 정규성을 확인합니다.
    일변량 정규성은 다변량 정규성(우리가 원하는 것)을 보장하지는 않지만 도움이됩니다.
    고려해야 할 또 다른 세부 사항은 큰 표본 (> 200 개의 관측치)에서 정규성이 문제가 되지 않는다는 것입니다.
    그러나 정규성을 해결하면 다른 많은 문제 (예 : 이질성)를 피할 수 있으므로 이 분석을 수행하는 주된 이유입니다.

등분산성 - 내가 제대로 썼으면 좋겠습니다. 등분산성이란 '종속변수(들)가 예측변수 범위에 걸쳐 동일한 수준의 분산을 보인다는 가정'을 의미합니다.(Hair et al., 2013).
    등분산성(homoscedasticity)은 모든 독립변수 값에서 오차 항이 동일하기를 원하기 때문에 바람직합니다.

선형성 - 선형성을 평가하는 가장 일반적인 방법은 산점도를 조사하고 선형 패턴을 검색하는 것입니다.
    패턴이 선형이 아닌 경우 데이터 변환을 탐색하는 것이 좋습니다.
    그러나 우리가 본 대부분의 산점도는 선형 관계를 갖는 것처럼 보이기 때문에 여기에 들어가지 않을 것입니다.

상관오차의 제거 - 정의에서 제시하는 것처럼 상관오차는 한 오차가 다른 오차와 상관 될 때 발생합니다. 
    예를 들어, 하나의 긍정적인 오차가 체계적으로 부정적인 오차를 만든다면 이는 이러한 변수 사이에 관계가 있음을 의미합니다.
    이것은 시계열에서 자주 발생하며 일부 패턴은 시간과 관련이 있습니다.
    우리는 이것에 대해서도 다루지 않을 것입니다.
    그러나 무언가를 감지하면 그 효과를 설명할 수 있는 변수를 추가하십시오.
    이것이 상관오차에 대한 가장 일반적인 솔루션입니다.

엘비스가이 긴 설명에 대해 뭐라고 말할까요?
'조금 덜 얘기하고, 좀 더 행동 해주세요'?
아마 ... 그건 그렇고, 엘비스의 마지막 히트작이 뭔지 아세요?

(...)

욕실 바닥.
'''

# In[ ]
'''
정규성을 찾기 위해
여기서 요점은 매우 간결한 방식으로 'SalePrice'를 테스트하는 것입니다.
다음 사항에 주의하여이 작업을 수행합니다.

히스토그램 - 첨도와 왜도.
정규 확률도 - 데이터 분포는 정규분포를 나타내는 대각선에 가깝게 따라야합니다.
'''

# In[ ]
#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm); # SalePrice에 대한 히스토그램과 데이터에 맞춘 정규분포를 그려 비교
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt) # stats.probplot : 앞의 그래프에 대한 Q-Q plot 작성

# In[ ]
'''
좋습니다. 'SalePrice'는 정상이 아닙니다.
'뾰족함'을 보이고, 양의 왜도를 나타내며 대각선을 따르지 않습니다.

그러나 모든 것이 손해는 아닙니다.
간단한 데이터 변환으로 문제를 해결할 수 있습니다.
이것은 통계학 책에서 배울 수있는 멋진 것 중 하나입니다.
양의 왜도가 있는 경우 로그 변환이 일반적으로 잘 작동합니다.
이것을 발견했을 때 나는 호그와트의 학생이 새로운 멋진 주문을 발견하는 것처럼 느꼈습니다.

Avada kedavra!
'''

# In[ ]
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice']) # 로그 변환/일반적으로는 log1p() 사용

# In[ ]
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

# In[ ]
'''
해결했습니다!
'GrLivArea'에서 무슨 일이 일어나고 있는지 확인합시다.
'''

# In[ ]
#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

# In[ ]
'''
뒤트린 맛... Avada kedavra!
'''

# In[ ]
#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

# In[ ]
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

# In[ ]
'''
Next, please...
'''

# In[ ]
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

# In[ ]
'''
좋아, 이제 우리는 큰 보스를 다루고 있습니다.
여기에 뭐가 있죠?

일반적으로 왜도를 나타내는 것.
값이 0 인 상당한 수의 관측치 (지하실이없는 주택).
0 값은 로그 변환을 허용하지 않기 때문에 큰 문제입니다.

여기에서 로그 변환을 적용하기 위해 지하실이 있거나 없는 효과를 얻을 수 있는 변수 (이진 변수)를 생성합니다.
그런 다음 0이 아닌 모든 관측 값에 대해 로그 변환을 수행하고 값이 0인 관측 값은 무시합니다.
이렇게하면 지하실의 유무에 따른 영향을 잃지 않고 데이터를 변환할 수 있습니다.

이 접근 방식이 올바른지 잘 모르겠습니다.
나에게 딱 맞는 것 같았습니다.
이것이 제가 '고위험 엔지니어링'이라고 부르는 것입니다.
'''

# In[ ]
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index) # TotalBsmtSF개수만큼 index별로 데이터 생성
df_train['HasBsmt'] = 0 # 생성된 데이터를 모두 0으로 바꿈
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1 # TotalBsmtSF이 양수인 데이터 1로 바꿈

# In[ ]
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF']) # HasBsmt가 1인 값에만 로그변환 값 입력

# In[ ]
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm); # TotalBsmtSF가 0인 값을 제외하고 그래프를 그림
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

# In[ ]
'''
첫 시도에 바로 '등분산성'를 적용 검색

두 개의 계량변수에 대한 등분산성을 테스트하는 가장 좋은 방법은 그래픽 방식입니다.
균등분산에서 이탈은 원뿔(그래프의 한쪽에 작은 분산, 반대쪽에 큰 분산) 또는 다이아몬드(분포 중심에 많은 수의 점)와 같은 모양으로 표시됩니다.

'SalePrice'및 'GrLivArea'로 시작 ...
'''

# In[ ]
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

# In[ ]
'''
이 산점도의 이전 버전(로그 변환 이전)은 원뿔 모양을 가졌습니다.
('SalePrice'와 상관변수 사이의 산포도 (move like Jagger style) 확인')
보시다시피, 현재 산점도는 더 이상 원뿔 모양이 아닙니다.
그것이 정규성의 힘입니다!
일부 변수의 정규성을 확인하여 등분산성 문제를 해결했습니다.

이제 'TotalBsmtSF'로 'SalePrice'를 확인하겠습니다.
'''

# In[ ]
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

# In[ ]
'''
일반적으로 'SalePrice'는 'TotalBsmtSF'범위에서 동일한 수준의 분산을 나타낸다고 말할 수 있습니다. 멋있습니다!
'''

# In[ ]
'''
마지막으로, 더미 변수
Easy 모드.
'''

# In[ ]
#convert categorical variable into dummy
df_train = pd.get_dummies(df_train) # 원핫인코딩 적용

# In[ ]
'''
결론
그게 다야! 연습이 끝났습니다.

이 커널 전반에 걸쳐 Hair et al.(2013)이 제안한 많은 전략을 실행했습니다.
우리는 변수에 대해 철학을 가지고 'SalePrice'만 분석하고
가장 상관관계가 있는 변수를 분석하고 결측치와 이상치를 처리하고
몇 가지 기본적인 통계 가정을 테스트했으며 범주형 변수를 더미 변수로 변환했습니다.
그것은 파이썬은 우리를 더 쉽게 만드는 데 도움이 된 많은 작업을 해주었습니다.

그러나 퀘스트는 끝나지 않았습니다.
우리의 이야기는 페이스북 조사에서 멈췄다는 것을 기억하십시오.
이제 'SalePrice'에 전화를 걸어 그녀를 저녁 식사에 초대 할 시간입니다.
그녀의 행동을 예측하십시오. 그녀가 정규화된 선형 회귀 접근 방식을 즐기는 소녀라고 생각하십니까?
아니면 그녀가 앙상블 방법을 선호한다고 생각합니까? 아니면 다른 것?

알아내는 것은 당신에게 달려 있습니다.
'''

# In[ ]
'''
References
My blog
My other kernels
Hair et al., 2013, Multivariate Data Analysis, 7th Edition

Acknowledgements
Thanks to João Rico for reading drafts of this.
'''
