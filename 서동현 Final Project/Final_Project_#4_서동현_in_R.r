
## Load Data
data2 <- read.table('data2.csv', header = TRUE, sep = ',', stringsAsFactors= T)

## party data
party <- read.csv('party.csv', header=T, sep = ',')
party_name = party[,1]
rownames(party) = party_name

# Row Name을 적용시켜 준다.
rowname = data2[,1]
data2 = data2[,-1]
row.names(data2) = rowname[]

head(data2)



data3 <- data.matrix(data2, rownames.force = NA)

head(data3)

# 공동 발의 건수의 의원에 따른 치우친 정도 판단을 위해 대각 원소를 행렬의 모든 원소로 나눠준다. 
data4 <- data3/diag(data3)

head(data4)

# bar plot을 그려 이상치를 확인한다.

par(mfrow = c(1, 1), oma=c(0,0,0,0))
boxplot(diag(data3))

##library setting
if(!require(cluster)){install.packages('cluster') ; library(cluster)}

hc.complete=hclust(dist(data4), method="complete") # 완전연결 (최대 클러스터 간 비유사성)
hc.average=hclust(dist(data4), method="average")   # 평균연결 (평균 클러스터 간 비유사성)
hc.single=hclust(dist(data4), method="single")     # 단일연결 (최소 클러스터 간 비유사성)

# Single, Complete, Average Clustering Dendrogram을 그려본다. 
par(mfrow = c(3, 1))
plot(hc.single, main = 'Single linkage Clustering') 
plot(hc.complete, main = 'Complete linkage Clustering') 
plot(hc.average, main = 'Average linkage Clustering')

# 그래프를 볼 때 적절한 군집의 수는 3~5개라고 판단이 되기에 3개의 군집으로 절단해 본다.
x <- cutree(hc.average, 3)

names(x)[x==1] 

names(x)[x==2] 

names(x)[x==3] 

data5 <- cbind(data.frame(x), party) 
data5$party[which(data5$x==1)]

data5$party[which(data5$x==2)]

data5$party[which(data5$x==3)] 

# Table을 이용하여 각 정당의 분포를 확인

table(data5$party[which(data5$x==1)])
table(data5$party[which(data5$x==2)])
table(data5$party[which(data5$x==3)])

# 군집 갯수 추가
x <- cutree(hc.average, 4)
data5 <- cbind(data.frame(x), party)

# Table을 이용하여 각 정당의 분포를 확인
table(data5$party[which(data5$x==1)])
table(data5$party[which(data5$x==2)])
table(data5$party[which(data5$x==3)])
table(data5$party[which(data5$x==4)])

# 4개의 군집으로 군집 분석을 수행
# 각 의원들의 군집 번호를 data.frame에 할당하고, 정당 정보를 결합

a <- pam(data4, 4) 

newdata <- cbind(data.frame(a$clustering), party) 
newdata <- newdata[,-2]
head(newdata)

# Table을 보여준다.
table(newdata$party[which(newdata$a.clustering==1)])
table(newdata$party[which(newdata$a.clustering==2)])
table(newdata$party[which(newdata$a.clustering==3)])
table(newdata$party[which(newdata$a.clustering==4)])

if(!require(ggfortify)){install.packages('ggfortify') ; library(ggfortify)}
autoplot(pam(data4, 4), frame = TRUE, frame.type = 'norm') # 군집의 수를 3~6으로 늘려가면서 관측하자

autoplot(pam(data4, 5), frame = TRUE, frame.type = 'norm')

autoplot(pam(data4, 6), frame = TRUE, frame.type = 'norm')

frame<- data2
nndata = NULL
for ( i in 1:141){
  for ( j in 1:141){
    if(data2[i,j] != data2[i,i]){
      nndata = data2[i,j]/sum(data2[i,-i])
      frame[i,j] = nndata
    }
  }
}

# 자기 자신에 해당하는 부분은 0으로 변환
frame1=frame
for ( i in 1:141){
  frame1[i,i]=0
}

# 매트릭스화
matrix1 <- data.matrix(frame1, rownames.force = NA)

# 행렬전치
tmp = NULL
for (i in 1:141)
{
  tmp = cbind(tmp, matrix1[i,])
}

colnames(tmp) = rownames(tmp)

# 당이름으로된 열 추가
tmp = cbind(party, tmp)
tmp = tmp[,-1] # id열 제거

tmp2 <- tmp[,-1] 
c = pam(tmp2, 6) 

c1 = c$clustering[c$clustering == 1]
c2 = c$clustering[c$clustering == 2]
c3 = c$clustering[c$clustering == 3]
c4 = c$clustering[c$clustering == 4]
c5 = c$clustering[c$clustering == 5]
c6 = c$clustering[c$clustering == 6]

cdata <- cbind(data.frame(c$clustering), party) # 정당 값들 결합
cdata2 <- cdata[,-2]
head(cdata2)

if(!require(factoextra)){install.packages('factoextra') ; library(factoextra)}

fviz_cluster(c)

# 각 클러스터에 속해있는 의원들의 정당 분포를 Table로 그려본다.

table(cdata2$party[which(cdata2$c.clustering==1)]) # 바른정당 6 / 자유한국당 23
table(cdata2$party[which(cdata2$c.clustering==2)]) # 더불어민주당 32
table(cdata2$party[which(cdata2$c.clustering==3)]) # 바른정당 1 / 자유한국당 18
table(cdata2$party[which(cdata2$c.clustering==4)]) # 국민의당 1 / 더불어민주당 21 / 무소속 1 
table(cdata2$party[which(cdata2$c.clustering==5)]) # 국민의당 18
table(cdata2$party[which(cdata2$c.clustering==6)]) # 국민의당 5 / 더불어민주당 14 / 정의당 1

library(ggplot2)

partyFreq = function(data, cluster)
{
  outPut = NULL
  for (i in 1:length(cluster))
  {
    partySum = c()
    g = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '국민의당')])
    d = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '더불어민주당')])
    f = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '자유한국당')])
    b = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '바른정당')])
    j = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '정의당')])
    m = sum(data[,colnames(data) == names(cluster[i])][which(data$party == '무소속')])
    partySum = c(g, d, f, b, j, m) 
    names(partySum) = c('국민의당', '더불어민주당', '자유한국당', '바른정당', '정의당', '무소속')
    
    outPut = rbind(outPut, partySum)
    rownames(outPut)[i] = names(cluster[i])
  }
  print('아래는 각 의원의 정당별 협업 정도 데이터프레임')
  print(outPut)
  result = colSums(outPut)/length(cluster)
  result2 = transform(result)
  result2 = cbind(rownames(result2), result2)
  colnames(result2) = c('Party', 'Proposition')
  pie<- ggplot(result2, aes(x='', y=Proposition, fill=Party))+
    geom_bar(width = 1, stat = "identity") + coord_polar("y", start = 0)
  plot(pie)
  return(result)
}

partyFreq(tmp, c1) #클러스터 1의 당 협업 정도

partyFreq(tmp, c2) #클러스터 2의 당 협업 정도

partyFreq(tmp, c3) #클러스터 3의 당 협업 정도

partyFreq(tmp, c4) #클러스터 4의 당 협업 정도

partyFreq(tmp, c5) #클러스터 5의 당 협업 정도

partyFreq(tmp, c6) #클러스터 6의 당 협업 정도

party <- read.csv('party.csv', header=T, sep = ',')
party_name = party[,1]
rownames(party) = party_name

fp = party[party$party=="자유한국당",]$id
bp = party[party$party=="바른정당",]$id
gp = party[party$party=="국민의당",]$id
dp = party[party$party=="더불어민주당",]$id
jp = party[party$party=="정의당",]$id
mp = party[party$party=="무소속",]$id

party = list(fp, bp, gp, dp, jp, mp) #당에 따른 의원 list에 할당
party_name = c('자유한국당', '바른정당', '국민의당', '더불어민주당', '정의당', '무소속')
party = setNames(party, party_name) #party 리스트이름 지정

# 협업/협업sum 비율척도로 바꾼다.
frame<- data2
nndata = NULL
for ( i in 1:141){
  for ( j in 1:141){
    if(data2[i,j] != data2[i,i]){
      nndata = data2[i,j]/sum(data2[i,-i])
      frame[i,j] = nndata
    }
  }
}
#자기 자신에 해당하는 부분은 0으로 변환
frame1=frame
for ( i in 1:141){
  frame1[i,i]=0
}

#매트릭스화
matrix1 <- data.matrix(frame1, rownames.force = NA)


# Divide by diag element
data4 <- matrix1
fp_data = data4[rownames(data4) %in% fp,!(colnames(data4) %in% fp)]
bp_data = data4[rownames(data4) %in% bp,!(colnames(data4) %in% bp)]
gp_data = data4[rownames(data4) %in% gp,!(colnames(data4) %in% gp)]
dp_data = data4[rownames(data4) %in% dp,!(colnames(data4) %in% dp)]
jp_data = data4[rownames(data4) %in% jp,!(colnames(data4) %in% jp)]
mp_data = data4[rownames(data4) %in% mp,!(colnames(data4) %in% mp)]

fp_pam = pam(fp_data, 2)
fviz_cluster(fp_pam)

#군집별 의원
fp_c1 = fp_pam$clustering[fp_pam$clustering == 1]
fp_c2 = fp_pam$clustering[fp_pam$clustering == 2]

topSeclect = function(data, cluster)
{
  outPut = c()
  for (i in 1:length(cluster))
  {
    allCong = data[rownames(data) == names(cluster[i]),]
    topThree = head(sort(allCong, decreasing = T), 3)
    print(topThree)
    outPut = c(outPut, names(topThree))
  }
  print(table(outPut))
  
}

topSeclect(fp_data, fp_c1)

topSeclect(fp_data, fp_c2)

dp_pam = pam(dp_data, 2)
fviz_cluster(dp_pam)

#군집별 의원
dp_c1 = dp_pam$clustering[dp_pam$clustering == 1]
dp_c2 = dp_pam$clustering[dp_pam$clustering == 2]

topSeclect(dp_data, dp_c1)

topSeclect(dp_data, dp_c2)

gp_pam = pam(gp_data, 2)
fviz_cluster(gp_pam)

#군집별 의원
gp_c1 = gp_pam$clustering[gp_pam$clustering == 1]
gp_c2 = gp_pam$clustering[gp_pam$clustering == 2]

topSeclect(gp_data, gp_c1)

topSeclect(gp_data, gp_c2)

bp_pam = pam(bp_data, 2)
fviz_cluster(bp_pam)

#군집별 의원
bp_c1 = bp_pam$clustering[bp_pam$clustering == 1]
bp_c2 = bp_pam$clustering[bp_pam$clustering == 2]
topSeclect(bp_data, bp_c1)

topSeclect(bp_data, bp_c2)




