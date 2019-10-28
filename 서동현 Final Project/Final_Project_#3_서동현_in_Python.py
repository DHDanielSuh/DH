
# coding: utf-8

# # ▣ 비즈니스 데이터마이닝 FINAL EXAM
# 
# ### 문제 3
# 
# ### 목표: 국회 의원들의 Ideo_self 값 예측

# ### 사용한 Data set: data1.csv
# 
# ### Ideo_self 에 NA 값이 있는 데이터를 Test set으로 보고 나머지 데이터를 Train set으로 보았다.  Train set을 통해 모델들을 학습시키고 학습한 모델을 토대로 Test set의 Ideo_self 값을 추정해 보았다. 
# 
# 다음 모델들의 테스트 성능을 비교해 본다.
# 1. Logistic regression
# 2. k-nearest neighbor classifier
# 3. naive Bayes classifier
# 4. Decision tree
# 5. Random forest
# 6. SVM
# 7. Xgboost
# 8. SoftMax
# 9. Keras + Relu + SoftMax

# # 0. Data preprocessing

# In[374]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[375]:


data = pd.read_excel('data1.xlsx', header = 1)


# In[376]:


data.head()


# ### ID column 앞에 있는 값들은 단순 index 값이므로 ID column을 새로운 index를 값으로 바꿔준다. 

# In[377]:


# data = data.set_index("id").sort_index()
data = data.set_index("id")


# In[378]:


data.head(10)


# ## Data 내에 모든 column 값이 NA인 값을 걸러준다. 

# In[379]:


data = data.dropna(how="all")


# In[380]:


data.shape


# ## 결과물을 원하는 모양으로 보여주기 위한 옵션을 건다. 

# In[381]:


pd.set_option('display.width', 100)  # 결과물을 잘 보여주기 위한 옵션, column 숫자를 표현한 것 같다. 10 단위로
pd.set_option('precision', 3)        # 결과물을 잘 보여주기 위한 옵션, 숫자 소수점 표현하는 것 
data.head()


# In[382]:


data.describe()


# In[383]:


data.dtypes


# In[384]:


data.info()


# In[385]:


data.isnull()


# In[ ]:





# ## NA 값들을 추측하기 위한 base 데이터를 잡아준다. 

# In[386]:


# k 질문들만 잡고 NA 값 추측
# x_impute = data.values[:,7:-1]

# 전체 데이터 잡고 NA값 추측 
x_impute = data.values[:,:-1]

pd.DataFrame(x_impute).isnull().any()


# In[387]:


x_impute.shape


# ## NA의 값으로 KNN, SoftImpute, SimpleFill, MICE 4개의 imputer를 쓸 예정이다.
# - **Manuel for KNN:** Nearest neighbor imputations which weights samples using the mean squared difference on features for which two rows both have observed data.
# - **Manuel for SimpleFill:** Replaces missing entries with the mean or median of each column.
# - **Thesis for SoftImpute:** [click] http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf
# - **Thesis for MICE:** [click] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
# 

# In[388]:


from fancyimpute import (
    KNN,
    SoftImpute,
    MICE
)


# ## MICE imputer로 missing Value를 처리했다. 

# In[389]:


x_impute_filled = MICE().complete(x_impute)


# In[390]:


x_impute_filled.shape


# In[391]:


pd.DataFrame(x_impute_filled).isnull().any()


# ## Data Set에서 Missing Value를 바꿀 colums들을 분류해준다. 

# In[392]:


array = data.values
# k 질문들만 잡고 NA 값 찾는 방법
# array[:, 7:-1].shape

# 전체 데이터를 잡고 NA값 찾는 방법
array[:,:-1].shape


# ## Impute 된 데이터를 Data Set에 적용시켜준다. 

# In[393]:


array[:, :-1] = x_impute_filled


# ## Ideo_self column을 제외한 본 Data Set에 더 이상 Missing Value 가 없다는 것을 확인할 수 있다. 

# In[394]:


pd.DataFrame(array).isnull().any()


# In[395]:


data.head(10)


# ## Impute 된 값이 확률로 나왔기 때문에 0.5 이상의 값은 1로 대체하고 0.5 이하의 값은 0으로 대체해준다. 

# In[396]:


for i in range(0, 1054):
    for j in range(0, 10):
        if np.any(array[:, 7:-1][i][j] >= 0.5):
            array[:, 7:-1][i][j] = 1
        else:
            array[:, 7:-1][i][j] = 0


# In[397]:


data.isnull().any()


# ## 확률로 된 Impute 값이 아닌 0 과 1 로 구성된 값을 볼 수 있다. 

# In[398]:


data.head()


# ## Ideo_self 가 class 별로 몇 개씩 있는지 확인해본다. 

# In[399]:


ideo_self_counts = data.groupby('ideo_self').size()
ideo_self_counts


# In[400]:


data.head()


# In[ ]:





# ## Encoding categorical variables
# - Categorical variable: 값이 nominal인 변수
# - 한 variable에 category가 총 C개가 존재하는 경우, 이를 C개의 binary dummy variables로 변환하여 수치형 데이터로 변환할 수 있다.
# - 모든 variable의 숫자 값이 category 값이기 때문에 **One hot encoding**을 통해 binary dummy variables로 변환해준다. 
# - Using `pandas.get_dummies`
#     - [click]: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
# 

# In[401]:


#from sklearn.preprocessing import OneHotEncoder
#encoder = OneHotEncoder()


# In[402]:


Sex_dummies = pd.get_dummies(data.sex, prefix = 'sex')
Sex_dummies.sample(n = 10)


# In[403]:


Age_dummies = pd.get_dummies(data.age, prefix = 'age')
Age_dummies.sample(n = 10)


# In[404]:


Area_dummies = pd.get_dummies(data.area, prefix = 'area')
Area_dummies.sample(n = 10)


# In[405]:


Edu_dummies = pd.get_dummies(data.edu, prefix = 'education')
Edu_dummies.sample(n = 10)


# In[406]:


Income_dummies = pd.get_dummies(data.income, prefix = 'income')
Income_dummies.sample(n = 10)


# In[407]:


k2_dummies = pd.get_dummies(data.k2, prefix = 'k2')
k2_dummies.sample(n = 10)


# In[408]:


k3_dummies = pd.get_dummies(data.k3, prefix = 'k3')
k3_dummies.sample(n = 10)


# In[409]:


k4_dummies = pd.get_dummies(data.k4, prefix = 'k4')
k4_dummies.sample(n = 10)


# In[410]:


k6_dummies = pd.get_dummies(data.k6, prefix = 'k6')
k6_dummies.sample(n = 10)


# In[411]:


k7_dummies = pd.get_dummies(data.k7, prefix = 'k7')
k7_dummies.sample(n = 10)


# In[412]:


k8_dummies = pd.get_dummies(data.k8, prefix = 'k8')
k8_dummies.sample(n = 10)


# In[413]:


k10_dummies = pd.get_dummies(data.k10, prefix = 'k10')
k10_dummies.sample(n = 10)


# In[414]:


k12_dummies = pd.get_dummies(data.k12, prefix = 'k12')
k12_dummies.sample(n = 10)


# In[415]:


k13_dummies = pd.get_dummies(data.k13, prefix = 'k13')
k13_dummies.sample(n = 10)


# In[416]:


k14_dummies = pd.get_dummies(data.k14, prefix = 'k14')
k14_dummies.sample(n = 10)


# ## Dummy Variable로 변경된 variables를 Data set에 추가해주고 그 파일 이름을 new_data라고 칭한다. 

# In[417]:


new_data = pd.concat([data,  Sex_dummies, Age_dummies, Area_dummies, Edu_dummies, Income_dummies, k2_dummies, k3_dummies, k4_dummies, k6_dummies, k7_dummies, k8_dummies, k10_dummies, k12_dummies, k13_dummies, k14_dummies], axis = 1)
new_data.head(10)


# ## Dummy variable로 이미 변경된 variables들은 더 이상 필요하지 않기 때문에 new_data set에서 제거해준다. 

# In[418]:


new_data = new_data.drop(['age', 'birth', 'area', 'sex', 'age1','edu', 'income','k2','k3','k4','k6','k7','k8','k10','k12','k13','k14'], axis = 1)
new_data.head(10)


# ## New_data에서 Ideo_self의 값이 NA인 row가 ID Number 24 부터이기 때문에 NA 값인 rows들을 따로 분류해준다. 

# In[419]:


new_data[24:]


# ## New_data의 ID Number 24 부터는 모두 NA 값이 포함되어 있는 것을 볼 수 있다. 

# In[420]:


new_data[24:].ideo_self.isnull()


# ## Ideo_self의 NA 값을 기준으로 Train data와 Test data를 나눠준다. 

# In[421]:


train_data = new_data[:199]
test_data = new_data[24:]


# In[422]:


train_data.head()


# In[423]:


test_data.head()


# In[ ]:





# ## Train Data를 X와 Y를 나눠주기 위해 Train data의 columns 이름과 갯수를 확인한다. 

# In[424]:


# 변수명 가져오기
col_names = train_data.columns.values


# In[425]:


# Ideo_self column은 첫 번째 column이다. 
col_names[:1]


# In[426]:


# Ideo_self를 기준으로 X와 Y를 나눠준다. 
train_X = train_data[col_names[1:]]
train_Y = train_data[col_names[:1]]


# In[427]:


train_X.head(5)


# In[428]:


train_Y.head()


# ## Test Data를 X와 Y를 나눠주기 위해 Test data의 columns 이름과 갯수를 확인한다. 

# In[429]:


# 변수명 가져오기
col_names = test_data.columns.values


# In[430]:


# Ideo_self column은 첫 번째 column이다. 
col_names[:1]


# In[431]:


# Ideo_self를 기준으로 X와 Y를 나눠준다. 
test_X = test_data[col_names[1:]]
test_Y = test_data[col_names[:1]]


# In[432]:


test_X.head(5)


# In[433]:


test_Y.head(5)


# In[ ]:





# # 1. Split Train data
# 1. Training set (70%)
# 2. Validation set (30%)
# 

# ## Model에 학습시키기 위해 Train set의 X와 Y 값을 array 값으로 바꿔준다.

# In[434]:


train_X = train_X.values
train_Y = train_Y.values


# In[435]:


# Y 값을 numpy.ravel 함수를 써서 reshape 시켜준다. Return a contiguous flattened array.

train_Y = np.ravel(train_Y)


# In[436]:


train_X


# In[437]:


train_Y


# ## Model에 학습시키기 위해 Test set의 X와 Y 값을 array 값으로 바꿔준다.

# In[438]:


test_X = test_X.values
test_Y = test_Y.values


# In[439]:


# Y 값을 numpy.ravel 함수를 써서 reshape 시켜준다. Return a contiguous flattened array.

test_Y = np.ravel(test_Y)


# In[440]:


test_X


# In[441]:


test_Y


# ## Skitlearn library를 통해 Train set을 Train 과 Validataion 으로 나눠준다. 

# In[442]:


from sklearn.model_selection import train_test_split


# In[443]:


train_X_train, train_X_val, train_Y_train, train_Y_val = train_test_split(train_X, train_Y, 
                                                        test_size=0.3, 
                                                        random_state=123)


# In[444]:


# Train, Validation Set의 shape을 확인해준다.

print(train_X_train.shape)
print(train_X_val.shape)
print(train_Y_train.shape)
print(train_Y_val.shape)


# In[ ]:





# # 2. Fit the model and compare validation AUCs and prediction probability
# 비교하고자 하는 classifiers들은 다음과 같음
# 1. Logistic regression
# 2. k-nearest neighbor classifier
# 3. naive Bayes classifier
# 4. Decision tree
# 5. Random forest
# 6. SVM
# 7. Xgboost
# 8. SoftMax

# In[79]:


from pprint import pprint
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc


# ## 2.1. Logistic regression
# Manual for `sklearn.linear_model.LogisticRegression`: [click](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 
# 다음 parameter들에 대해 validation data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다. 
# 1. penalty
# 2. C

# In[80]:


# C가 클수록 weak regularization
penalty_set = ['l1', 'l2']
C_set = [0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]


# In[81]:


result1 = []
for penalty in penalty_set:
    for C in C_set:
        logreg_model = LogisticRegression(penalty=penalty, C=C, class_weight='balanced', multi_class="multinomial", solver='saga', max_iter=10000)
        logreg_model = logreg_model.fit(train_X_train, train_Y_train)
#         Y_val_score = model.decision_function(train_X_val)
        Y_val_score = logreg_model.predict_proba(train_X_val)[:, 1]
        val_proba = "{:.4f}".format(logreg_model.score(train_X_val, train_Y_val))
        fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
        result1.append((logreg_model, penalty, C, val_proba, auc(fpr, tpr)))


# In[82]:


result1


# In[83]:


logreg_result = sorted(result1, key=lambda x: x[3], reverse=True)


# In[84]:


logreg_result


# ## Best Result에 대해 보여준다. 

# In[85]:


best_logreg_result = logreg_result[0]
print(best_logreg_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[86]:


best_logreg_model = best_logreg_result[0]
best_logreg_model = best_logreg_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_logreg_model.predict(train_X_val), train_Y_val))


# In[87]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_logreg_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_logreg_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[88]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_logreg_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_logreg_model.predict(train_X_val)))


# In[146]:


print("훈련 데이터에 있는 클래스 종류: {}".format(best_logreg_model.classes_))
argmax_dec_func = np.argmax(best_logreg_model.decision_function(train_X_train), axis=1)
print("가장 큰 결정 함수의 인덱스: {}".format(argmax_dec_func[:10]))
print("인덱스를 classses_에 연결: {}".format(best_logreg_model.classes_[argmax_dec_func][:10]))
print("Validation set의 예측: {}".format(best_knn_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_logreg_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_logreg_model.predict(test_X)[:10]))


# In[447]:


print("Test set의 전체 예측: {}".format(best_logreg_model.predict(test_X)))


# ## 2.2. k-nearest neighbor classifier
# Manual for `sklearn.neighbors.KNeighborsClassifier`: [click](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
# 
# 다음 parameter들에 대해 validation data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다. 
# 1. n_neighbors
# 2. weights

# In[92]:


weights_set = ['uniform', 'distance']
n_neighbors_set = [1, 3, 5, 7, 9, 11, 13, 15]


# In[94]:


result2 = []
for weights in weights_set:
    for n_neighbors in n_neighbors_set:
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        knn_model = knn_model.fit(train_X_train, train_Y_train)
        Y_val_score = knn_model.predict_proba(train_X_val)[:, 1]
        val_proba = "{:.4f}".format(knn_model.score(train_X_val, train_Y_val))
        fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
        #result.append("모델: {}, 최적 Weight: {}, 최적 N_neighbor: {}, 최적 AUC 값: {}".format(model, weights, n_neighbors, auc(fpr, tpr)))
        #result.append(("모델", model, "최적 Weigth", weights, "최적 N_neighbor", n_neighbors, "최적 AUC 값", auc(fpr, tpr)))       
        result2.append((knn_model, weights, n_neighbors, val_proba, auc(fpr, tpr)))        


# In[95]:


result2


# In[96]:


knn_result = sorted(result2, key=lambda x: x[3], reverse=True)


# In[97]:


knn_result


# ## Best Result에 대해 보여준다.

# In[98]:


best_knn_result = knn_result[0]
print(best_knn_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[99]:


best_knn_model = best_knn_result[0]
best_knn_model = best_knn_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_knn_model.predict(train_X_val), train_Y_val))


# In[100]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_knn_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_knn_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[101]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_knn_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_knn_model.predict(train_X_val)))


# In[149]:


# KNN 에는 decision function이 없어 predict_proba만 실행한다.
print("훈련 데이터에 있는 클래스 종류: {}".format(best_knn_model.classes_))
print("Validation set의 예측: {}".format(best_knn_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_knn_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_knn_model.predict(test_X)[:10]))


# In[448]:


print("Test set의 전체 예측: {}".format(best_knn_model.predict(test_X)))


# In[ ]:





# ## 2.3. Naive Bayes classifier
# Manual for `sklearn.naive_bayes.GaussianNB`: [click](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
# 
# 클래스에 대한 prior 정보를 조절하여 fitting

# In[110]:


priors_set = [None]


# In[112]:


result3 = []
for priors in priors_set:
    nb_model = GaussianNB(priors=priors)
    nb_model = nb_model.fit(train_X_train, train_Y_train)
    Y_val_score = nb_model.predict_proba(train_X_val)[:, 1]
    val_proba = "{:.4f}".format(nb_model.score(train_X_val, train_Y_val))
    fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
    result3.append((nb_model, priors, val_proba, auc(fpr, tpr)))      


# In[113]:


nb_result = sorted(result3, key=lambda x: x[2], reverse=True)


# In[114]:


nb_result


# ## Best Result를 보여준다.

# In[115]:


best_nb_result = nb_result[0]
print(best_nb_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[116]:


best_nb_model = best_nb_result[0]
best_nb_model = best_nb_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_nb_model.predict(train_X_val), train_Y_val))


# In[117]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_nb_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_nb_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[118]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_nb_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_nb_model.predict(train_X_val)))


# In[148]:


# GaussianNB는 Decision Function이 없어 predict_proba만 실행한다.
print("훈련 데이터에 있는 클래스 종류: {}".format(best_nb_model.classes_))
print("Validation set의 예측: {}".format(best_nb_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_nb_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_nb_model.predict(test_X)[:10]))


# In[449]:


print("Test set의 전체 예측: {}".format(best_nb_model.predict(test_X)))


# In[ ]:





# ## 2.4. Decision tree
# Manual for `sklearn.tree.DecisionTreeClassifier`: [click](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
# 
# 다음 parameter들에 대해 validation data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다. 
# 1. max_depth
# 2. class_weight

# In[122]:


class_weight_set = [None, 'balanced']
max_depth_set = [3, 4, 5, 6, 7]


# In[123]:


result4 = []

for class_weight in class_weight_set:
    for max_depth in max_depth_set:
        dt_model = DecisionTreeClassifier(class_weight=class_weight, max_depth=max_depth)
        dt_model = dt_model.fit(train_X_train, train_Y_train)
        Y_val_score = dt_model.predict_proba(train_X_val)[:, 1]
        val_proba = "{:.4f}".format(dt_model.score(train_X_val, train_Y_val))
        fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
        result4.append((dt_model, class_weight, max_depth, val_proba, auc(fpr, tpr)))


# In[124]:


dt_result = sorted(result4, key=lambda x: x[3], reverse=True)


# In[125]:


dt_result


# ## Best Result에 대해 보여준다. 

# In[126]:


best_dt_result = dt_result[0]
print(best_dt_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[127]:


best_dt_model = best_dt_result[0]
best_dt_model = best_dt_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_dt_model.predict(train_X_val), train_Y_val))


# In[128]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_dt_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_dt_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[129]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_dt_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_dt_model.predict(train_X_val)))


# In[150]:


# Decision Tree Classifier은 Decision Function이 없어 predict_proba만 실행한다. 
print("훈련 데이터에 있는 클래스 종류: {}".format(best_dt_model.classes_))
print("Validation set의 예측: {}".format(best_dt_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_dt_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_dt_model.predict(test_X)[:10]))


# In[450]:


print("Test set의 전체 예측: {}".format(best_dt_model.predict(test_X)))


# In[ ]:





# ## 2.5. Random forest
# Manual for `sklearn.ensemble.RandomForestClassifier`: [click](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
# 
# 다음 parameter들에 대해 validation data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다. 
# 1. n_estimators
# 2. max_features

# In[365]:


n_estimators_set = [40, 60, 90, 100, 1000, 2000, 5000]
max_depth_set = [3, 4, 5, 6, 7]
max_features_set = ['auto', 'sqrt', 'log2']


# In[366]:


result5 = []
for n_estimators in n_estimators_set:
    for max_features in max_features_set:
        for max_depth in max_depth_set:
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=0)
            rf_model = rf_model.fit(train_X_train, train_Y_train)
            Y_val_score = rf_model.predict_proba(train_X_val)[:, 1]
            val_proba = "{:.4f}".format(rf_model.score(train_X_val, train_Y_val))
            fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
            result5.append((rf_model, n_estimators, max_features, max_depth, val_proba, auc(fpr, tpr)))


# In[367]:


rf_result = sorted(result5, key=lambda x: x[4], reverse=True)


# In[368]:


rf_result


# ## Best Result에 대해 보여준다 

# In[369]:


best_rf_result = rf_result[0]
print(best_rf_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[370]:


best_rf_model = best_rf_result[0]
best_rf_model = best_rf_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_rf_model.predict(train_X_val), train_Y_val))


# In[371]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_rf_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_rf_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[372]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_rf_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_rf_model.predict(train_X_val)))


# In[445]:


# Random Forest Classifier은 Decision Function이 없어 predict_proba만 실행한다. 
print("훈련 데이터에 있는 클래스 종류: {}".format(best_rf_model.classes_))
print("Validation set의 예측: {}".format(best_rf_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_rf_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_logreg_model.predict(test_X)[:10]))


# In[452]:


print("Test set의 전체 예측: {}".format(best_logreg_model.predict(test_X)))


# In[ ]:





# ## 2.6. SVM
# 
# Manual for `Support Vector Machines`: [click](http://scikit-learn.org/stable/modules/svm.html)
# 
# 다음 parameter들에 대해 validation data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다. 
# 1. gamma
# 2. C

# In[161]:


gamma_set = [0.001, 0.01, 0.1, 1, 10, 100]
c_set = [0.001, 0.01, 0.1, 1, 10, 100]


# In[162]:


result6 = []
for gamma in gamma_set:
    for C in c_set:
        svm_model = SVC(decision_function_shape='ovo', gamma=gamma, C=C, probability = True, max_iter=10000)
        svm_model = svm_model.fit(train_X_train, train_Y_train)
        Y_val_score = svm_model.predict_proba(train_X_val)[:, 1]
        val_proba = "{:.4f}".format(svm_model.score(train_X_val, train_Y_val))
        fpr, tpr, _ = roc_curve(train_Y_val, Y_val_score, pos_label=True)
        result6.append((svm_model, gamma, C,val_proba, auc(fpr, tpr)))


# In[163]:


svm_result = sorted(result6, key=lambda x: x[3], reverse=True)


# In[164]:


svm_result


# ## Best Result에 대해 보여준다.

# In[165]:


best_svm_result = svm_result[0]
print(best_svm_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[166]:


best_svm_model = best_svm_result[0]
best_svm_model = best_svm_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_svm_model.predict(train_X_val), train_Y_val))


# In[167]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_svm_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_svm_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[168]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_svm_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_svm_model.predict(train_X_val)))


# In[171]:


print("훈련 데이터에 있는 클래스 종류: {}".format(best_svm_model.classes_))
argmax_dec_func = np.argmax(best_svm_model.decision_function(train_X_val), axis=1)
print("가장 큰 결정 함수의 인덱스: {}".format(argmax_dec_func[:10]))
print("Validation set의 예측: {}".format(best_svm_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_svm_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_logreg_model.predict(test_X)[:10]))


# In[453]:


print("Test set의 예측: {}".format(best_logreg_model.predict(test_X)))


# In[ ]:





# ## 2.7. Xgboost
# 
# Manual for `Xgboost`: [click](http://xgboost.readthedocs.io/en/latest/how_to/param_tuning.html)
# 
# 다음 parameter들에 대해 data에 대한 Score값과 AUC값을 이용해 최적 모형 parameter를 찾는다.
# 
# 1. max_depth
# 2. min_child_weight
# 3. gamma
# 4. subsample
# 5. colsample_bytree
# 6. reg_alpha 

# In[173]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


# ## Grid Search CV 를 통해 parameter들을 최적의 값으로 조정해준다. 

# In[174]:


# max_depth와 min_child_weight 범위 설정을 통해 최적 parameter 값을 찾는다. 

param_test1 = {
 'max_depth':range(1,20),
 'min_child_weight':range(1,20)
}


# In[175]:


# Grid Search를 통해 param_test1에 대한 최적 값을 찾는다. 

grid_search1 = GridSearchCV(XGBClassifier(), param_test1, cv=5)
grid_search1.fit(train_X_train, train_Y_train)


# In[192]:


print("최적 매개변수: {}".format(grid_search1.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search1.best_score_))


# In[177]:


# gamma 범위 설정을 통해 최적 parameter 값을 찾는다. 

param_test2 = {
 'gamma': [i/100.0 for i in range(0,5)]
}


# In[178]:


# Grid Search를 통해 param_test2에 대한 최적 값을 찾는다. 

grid_search2 = GridSearchCV(XGBClassifier(max_depth = 2, min_child_weight= 8), param_test2, cv=5)
grid_search2.fit(train_X_train, train_Y_train)


# In[191]:


print("최적 매개변수: {}".format(grid_search2.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search2.best_score_))


# In[180]:


# subsample 과 colsample_bytree 범위 설정을 통해 최적 parameter 값을 찾는다. 

param_test3 = {
 'subsample':[i/100.0 for i in range(6,10)],
 'colsample_bytree':[i/100.0 for i in range(6,10)]
}


# In[181]:


# Grid Search를 통해 param_test3에 대한 최적 값을 찾는다. 

grid_search3 = GridSearchCV(XGBClassifier(gamma = 0.01, max_depth = 2, min_child_weight= 8), param_test3, cv=5)
grid_search3.fit(train_X_train, train_Y_train)


# In[190]:


print("최적 매개변수: {}".format(grid_search3.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search3.best_score_))


# In[183]:


# reg_alpha 범위 설정을 통해 최적 parameter 값을 찾는다. 

param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100, 0, 0.001, 0.005, 0.01, 0.05]
}


# In[184]:


# Grid Search를 통해 param_test4에 대한 최적 값을 찾는다. 

grid_search4 = GridSearchCV(XGBClassifier(colsample_bytree = 0.06, subsample = 0.09, gamma = 0.01, max_depth = 2, min_child_weight= 8), param_test4, cv=5)
grid_search4.fit(train_X_train, train_Y_train)


# In[189]:


print("최적 매개변수: {}".format(grid_search4.best_params_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search4.best_score_))


# In[186]:


# 최적 parameter 값을 이용해 Grid Search를 한 번 더 한다. 

param_test5 = {
 'reg_alpha':[1e-05], 
 'colsample_bytree':[0.06], 
 'subsample':[0.09], 
 'gamma':[0.01], 
 'max_depth':[2],
 'min_child_weight':[8]
}


# In[187]:


grid_search = GridSearchCV(XGBClassifier(reg_alpha = 1e-05, colsample_bytree = 0.06, subsample = 0.09, gamma = 0.01, max_depth = 2, min_child_weight= 8), param_test5, cv=5)
grid_search.fit(train_X_train, train_Y_train)


# In[188]:


print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))
print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))


# ## Xgboost Classifier에 최적 parameter 값들을  apply해 fitting 시킨다. 

# In[193]:


xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=2,
 min_child_weight=8,
 gamma=0.01,
 subsample=0.09,
 colsample_bytree=0.06,
 reg_alpha = 1e-05,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27) 


# In[194]:


xgb1.fit(train_X_train, train_Y_train)


# In[195]:


from sklearn.metrics import mean_squared_error


# ## MSE 값을 측정한다. 

# In[196]:


mean_squared_error(train_Y_val, xgb1.predict(train_X_val))


# In[197]:


get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# ## Feature Importance를 그래프로 그려 확인한다. 

# In[198]:


xgb.plot_importance(xgb1)


# In[201]:


#Predict training set:
dtrain_predictions = xgb1.predict(train_X_train)
dtrain_predprob = xgb1.predict_proba(train_X_train)
#Print model report:
print("\nModel Report")
print("Accuracy : %.4g" % metrics.accuracy_score(train_Y_train, dtrain_predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(train_X_train, dtrain_predprob))


# ## 성능을 좀 더 올리기 위해 Learning rate을 낮추고 n_estimator를 통해 tree를 더 추가한다 

# In[202]:


xgb2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=10000,
 max_depth=2,
 min_child_weight=8,
 gamma=0.01,
 subsample=0.09,
 colsample_bytree=0.06,
 reg_alpha = 1e-05,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27) 


# In[203]:


xgb2.fit(train_X_train, train_Y_train)


# ## MSE 값을 측정한다. 

# In[204]:


mean_squared_error(train_Y_val, xgb2.predict(train_X_val))


# ## Feature Importance를 그래프로 그려 확인한다. 

# In[205]:


xgb.plot_importance(xgb2)


# In[216]:


#Predict training set:
dtrain_predictions = xgb2.predict(train_X_val)
dtrain_predprob = xgb2.predict_proba(train_X_val)
#Print model report:
print("\nModel Report")
print("Accuracy : %.4g" % metrics.accuracy_score(train_Y_val, dtrain_predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(train_X_val, dtrain_predprob))


# In[218]:


print("Validation set의 예측: {}".format(xgb2.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(xgb2.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(xgb2.predict(test_X)[:10]))


# ## Best Result에 대해 보여준다. 

# In[263]:


best_xgb_result = xgb2
print(best_xgb_result)


# ## Best Model의 MAE 값을 보여준다. 

# In[357]:


best_xgb_model = best_xgb_result
best_xgb_model = best_xgb_model.fit(train_X_train, train_Y_train)
print(metrics.mean_absolute_error(best_xgb_model.predict(train_X_val), train_Y_val))


# In[266]:


# predict_proba 결과 중 앞부분 6개에 대해서만 확인한다.
print("예측 확률:\n{}".format(best_xgb_model.predict_proba(train_X_val)[:6]))

# 행 방향으로 확률을 더하면 모두 1이 된다.
print("합: {}".format(best_xgb_model.predict_proba(train_X_val)[:6].sum(axis=1)))


# ## predict_proba의 결과에 argmax 함수를 적용해서 예측을 재연할 수 있다. 

# In[267]:


print("가장 큰 예측 확률의 인덱스:\n{}".format(np.argmax(best_xgb_model.predict_proba(train_X_val), axis=1)))
print("예측:\n{}".format(best_xgb_model.predict(train_X_val)))


# In[269]:


print("훈련 데이터에 있는 클래스 종류: {}".format(best_xgb_model.classes_))
print("Validation set의 예측: {}".format(best_xgb_model.predict(train_X_val)[:10]))
print("실제 Validation set: {}".format(train_Y_val[:10]))
print("Validation Set의 정확도: {:.2f}".format(best_xgb_model.score(train_X_val, train_Y_val)))
print("Test set의 예측: {}".format(best_xgb_model.predict(test_X)[:10]))


# In[454]:


print("Test set의 전체 예측: {}".format(best_xgb_model.predict(test_X)))


# In[ ]:





# ## 2.8 SoftMax
# 
# Manual for `Soft Max`: [click](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
# 
# 
# SoftMax 모델을 통해 X와 Y variable들을 One Hot Encoding 해주고 최적 Accuracy를 찾아낸다. 
# 

# In[270]:


import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility


# In[271]:


# Predicting animal type based on various features
xy = np.loadtxt('new_data.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# In[272]:


print(x_data.shape, y_data.shape)


# ## Y 값의 클래스가 11개이므로 nb_classes를 11개로 잡는다.

# In[273]:


nb_classes = 11  # 0 ~ 10


# In[274]:


X = tf.placeholder(tf.float32, [None, 15])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 10


# In[275]:


Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)


# In[276]:


Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)


# In[277]:


W = tf.Variable(tf.random_normal([15, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


# In[278]:


# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)


# In[279]:


# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[304]:


# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result8=[]
    for step in range(50000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            result8.append([hypothesis, step, loss, acc])
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


# In[305]:


sm_result = sorted(result8, key=lambda x: x[3], reverse=True)


# In[306]:


sm_result


# ## Best Result에 대해 보여준다. 

# In[315]:


best_sm_result = sm_result[:1]
print(best_sm_result)


# In[325]:


test_X.shape


# ## Test set의 예측한 Y 값을 보여준다. 

# In[343]:


test_X = np.loadtxt('test_X.csv', delimiter=',', dtype=np.float32)
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: test_X})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p in pred:
        print("Prediction: {} ".format(p))


# In[ ]:





# ## 2.9 Keras + Relu + SoftMax
# 
# Description for `Keras + Relu + SoftMax`: 
# [click](https://keras.io/)
# [click](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions)
# [click](https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/)
# 
# Keras를 통해 고속 구현을 하고  Relu Activation Function을 통해 hidden layer에서 값을 찾아내 RNN을 구현한 후 SoftMax로 결과값을 처리해준 것이다. 

# In[344]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping


# In[345]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
1
2
3
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[346]:


# load dataset
dataframe = pandas.read_csv("new_data.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:15].astype(float)
Y = dataset[:,15]
dataframe2 = pandas.read_csv("test_X.csv", header = None)
dataset2 = dataframe2.values
test_x = dataset2[:,0:15]


# In[347]:


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[348]:


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=15, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[349]:


estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=5, verbose=0)


# In[350]:


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


# In[351]:


results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[352]:


# create model
model = Sequential()
model.add(Dense(100, input_dim=15, activation='relu'))
model.add(Dense(11, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[353]:


history = model.fit(X,dummy_y,epochs = 1000)


# ## Train Data를 토대로 구한 Prediction 예측 값이다. 

# In[358]:


model.predict_classes(X)


# In[361]:


model.predict_proba(X)


# ## Test set의 예측값이다.
# 

# In[360]:


y_pred = model.predict_classes(test_x)


# In[355]:


y_pred


# In[ ]:





# # 3. Conclusion

# ## - Data 전처리 과정
# 
# ### Data set의 모든 변수가 categorical number이기 때문에 One Hot Encoding을 통해 numerical 의미를 같는 dummy 변수로 바꿔주어야 했다. 
# ### K2 ~ K14까지의 질문들에 Missing Value들은 MICE 모델을 통해 Imputation 해주었다. SoftImpute과 KNN 방법을 써 동일한 모델을 돌려보았으나 성능 차이가 크게 나지 않아 MICE Imputation을 썼다. 
# ### Ideo_self column의 Missing Value 유무에 따라 Train set과 Test set으로 나눠주었다.  
# 
# ## - Data Modeling 과정
# 
# ### Train set을 토대로 위의 9개의 모델들의 성능을 평가해 최적 모델의 정확도와 AUC 값은 다음과 같다. 
# 
# ### ***`Parameter 조정으로 찾은 최적 Logistic Classifier의 결과값`***
#     - MAE: 2.607
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - 가장 큰 결정 함수의 인덱스: [ 0  1 10  8  0  6  9  1  2  3]
#     - 인덱스를 classses_에 연결: [  0.   1.  10.   8.   0.   6.   9.   1.   2.   3.]
#     - Validation set의 예측(상위 10개): [  5.   5.   5.   3.  10.  10.   4.   5.  10.   5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.14
#     - Test set의 예측(상위 10개): [  8.  10.   1.   8.   7.  10.   8.   0.   1.   9.]
# 
# ### ***`Parameter 조정으로 찾은 최적 KNN의 결과값`***
#     - MAE:1.445
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [  5.   5.   5.   3.  10.  10.   4.   5.  10.   5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.30
#     - Test set의 예측(상위 10개): [ 8.  8.  5.  8.  5.  8.  6.  5.  5.  9.]
# 
# ### ***`Parameter 조정으로 찾은 최적 Naive Bayes classifier의 결과값`***
#     - MAE: 3.633
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [  4.   1.  10.   7.  10.   9.   0.   1.  10.   1.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.04
#     - Test set의 예측(상위 10개): [  9.  10.   1.   1.   9.   9.   9.   1.   1.   9.]
# 
# ### ***`Parameter 조정으로 찾은 최적 Decision Tree의 결과값`***
#     - MAE: 1.611
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [  4.   5.   5.   5.   5.   8.   5.   3.  10.   5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.29
#     - Test set의 예측(상위 10개): [  8.  10.   3.   8.   5.   8.   8.   3.   5.   8.]
# 
# ### ***`Parameter 조정으로 찾은 최적 Random Forest의 결과값`***
#     - MAE: 1.478
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [ 5.  5.  5.  5.  5.  8.  5.  5.  5.  5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.35
#     - Test set의 예측(상위 10개): [  8.  10.   1.   8.   7.  10.   8.   0.   1.   9.]
# 
# ### ***`Parameter 조정으로 찾은 최적 SVM의 결과값`***
#     - MAE: 1.415
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - 가장 큰 결정 함수의 인덱스: [31 40 38 43 50 52 43 43 41 43]
#     - Validation set의 예측(상위 10개): [ 5.  5.  5.  5.  5.  8.  5.  5.  5.  5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.34
#     - Test set의 예측(상위 10개): [  8.  10.   1.   8.   7.  10.   8.   0.   1.   9.]
# 
# ### ***`Parameter 조정으로 찾은 최적 Xgboost의 결과값`***
#     - MAE: 1.548
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [ 5.  5.  5.  5.  5.  5.  5.  5.  5.  5.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.33
#     - Test set의 예측(상위 10개): [ 4.  5.  5.  5.  5.  5.  5.  5.  5.  5.]
#     
# ### ***`Parameter 조정으로 찾은 최적 SoftMax의 결과값`***
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [ 5.  7.  5.  6.  5.  3.  5.  5.  5.  9.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.37
#     - Test set의 예측(상위 10개): [ 3.  3.  4.  4.  4.  4.  3.  4.  4.  3.]
# 
# ### ***`Parameter 조정으로 찾은 최적 Keras + Relu + SoftMax의 결과값`***
#     - 훈련 데이터에 있는 클래스 종류: [  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.]
#     - Validation set의 예측(상위 10개): [ 5.  7. 10.  3.  6.  5.  5.  6.  5.  9.]
#     - 실제 Validation set(상위 10개): [  0.   7.   7.   5.  10.  10.   5.   5.   7.   5.]
#     - Validation Set의 정확도: 0.32
#     - Test set의 예측(상위 10개): [ 10.  5.  1.  8.  5.  7.  7.  5.  5.  6.] 
# 
# 
# ## - Data Interpretation 
# 
# ### 9개의 모델을 돌려본 결과 지도 학습 모델에서는 Random Forest, SVM, Xgboost가 가장 좋은 Validation set 정확도를 보여줬다. 비지도 학습 모델에서는 SoftMax 모델이 37%의 Validation set 정확도를 보여줌으로써 9개의 모델 중 가장 좋은 결과물을 보여줬다.  
# ### SoftMax 모델을 쓴 이유는 일단 모든 변수들이 Categorical value이기에 One Hot Encoding을 해 numerical type으로 바꿔줄 필요가 있었다. Softmax 모델의 입력받은 값은  0~1사이의 값으로 모두 정규화되며 출력 값들의 총합은 항상 1이 되는 특성을 가진다. 출력은 분류하고 싶은 클래수의 수 만큼 구성되며 가장 큰 출력 값을 부여받은 클래스가 확률이 가장 높은 것으로 이용했다. 그러나 성능이 생각보다 높게 나오지 않았기 때문에 보완의 필요성을 느꼈다. 
# ### SoftMax 모델을 보완하기 위해 Keras를 통해 고속 계산을 하고 Relu Activation Function을 사용해 hidden layer 값을 구해 SoftMax 모델에 적용시켜 보았다. 그러나 오히려 SoftMax 모델 하나만을 사용한 것보다 성능이 떨어지는 결과물을 보여줬다. 
# ### 각 모델의 Parameter들을 Grid Search를 통해 조절해가며 최적 모델을 찾아 예측을 해보았지만 데이터가 총 1000개 정도 밖에 되지 않기에 더 이상의 성능을 끌어올릴 수 없었다. Parameter들을 조금 더 세밀하게 수정한다면 약간의 상승 효과는 볼 수 있을 것으로 예상되나 그 차이가 미미할 것으로 보이기에 더 이상 시도하지 않았다. 더 많은 데이터 셋을 이용해 모델링을 해본다면 분명 더 높은 성능을 보일 것이라고 생각된다. 
# 
# 
# ## 결과 비교
# 
# ### 실제 데이터와 예측 값을 비교하려면 위의 모델링마다 마지막에 test set에 의한 예측값을 뽑아놨다.
# 
# 
# 

# In[ ]:




