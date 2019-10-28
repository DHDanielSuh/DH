


### Number2

## Library

library(ISLR)
library(MASS)  
library(class) # Classification

## Data Load

Auto=na.omit(Auto) # Auto 데이터 셋의 결측치 제거
summary(Auto)
str(Auto)
attach(Auto)

## 1. mpg01 변수 생성 

mpg01 = rep(0, length(mpg))  
mpg01[mpg > median(mpg)] = 1 # median 보다 큰 값은 1을 부여
Auto = data.frame(Auto, mpg01)
head(Auto)

## 2. LDA 수행 (짝수년 기준)

train = (year%%2 == 0)  # 짝수년을 train으로 할당
test = !train # 홀수년을 test으로 할당
Auto.train = Auto[train, ] # train set 생성
Auto.test = Auto[test, ] # test set 생성
mpg01.test = mpg01[test] 

lda.fit = lda(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, subset = train) # LDA 모형 적합
lda.fit

lda.pred = predict(lda.fit, Auto.test) # 예측값 생성
lda.pred
names(lda.pred)

lda.class=lda.pred$class # 예측 분류 값을 할당

table(lda.class,mpg01.test) # 혼동 행렬을 생성하여 예측값과 test set 비교
mean(lda.class==mpg01.test) 
mean(lda.class != mpg01.test)
(14+9)/(182) # 전체 오차율은 12.6%



## 3. QDA 수행

qda.fit = qda(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, subset = train) # QDA 모형 적합
qda.pred = predict(qda.fit, Auto.test) # 예측값 생성

qda.class=qda.pred$class  # 예측 분류 값 할당
mean(qda.pred$class != mpg01.test)
mean(qda.class==mpg01.test)
table(qda.class,mpg01.test)
(13+11)/(182) # 전체 오차율은 13.2%

## 4. 다중 로지스틱 회귀 분석 수행

glm.fit = glm(mpg01 ~ cylinders + weight + displacement + horsepower, data = Auto, family = binomial, subset = train) # 다중 로지스틱 회귀 모형 적합
glm.probs = predict(glm.fit, Auto.test, type = "response") # 예측 확률 생성
glm.pred = rep(0, length(glm.probs))
glm.pred[glm.probs > 0.5] = 1 # 확률이 0.5보다 크면 1로 분류, 아니면 0으로 분류

table(glm.pred, mpg01.test)
mean(glm.pred == mpg01.test)
mean(glm.pred != mpg01.test) 
(11+11)/(182) # 전체 오차율은 12.1%

## 5. KNN 수행

train.x = cbind(cylinders, weight, displacement, horsepower)[train, ] # train 데이터 set 설정(짝수년도)
test.x = cbind(cylinders, weight, displacement, horsepower)[test, ] # test 데이터 set 설정 (홀수년도)

train.mpg01 = mpg01[train] # mpg01 중 짝수년도에 해당하는 data set을 train set으로 설정

set.seed(1)

knn.pred = knn(train.x, test.x, train.mpg01, k = 1) #k가 1인 KNN 수행
mean(knn.pred != mpg01.test) #k가 1일때의 전체오차율은 15.4%

knn.pred = knn(train.x, test.x, train.mpg01, k = 10) #k가 10인 KNN 수행
mean(knn.pred != mpg01.test) #k가 10일때의 전체오차율은 16.5%

knn.pred = knn(train.x, test.x, train.mpg01, k = 100) #k가 100인 KNN 수행
mean(knn.pred != mpg01.test) #k가 100일때의 전체오차율은 14.3%
