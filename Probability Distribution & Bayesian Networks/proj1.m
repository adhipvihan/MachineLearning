data=xlsread('Universitydata.xlsx')
data=data(:,[3,4,5,6])
a=data(:,1)
b=data(:,2)
c=data(:,3)
d=data(:,4)
mu1=mean(a)
mu2=mean(b)
mu3=mean(c)
mu4=mean(d)
sigma1=std(a)
sigma2=std(b)
sigma3=std(c)
sigma4=std(d)
var1=var(a)
var2=var(b)
var3=var(c)
var4=var(d)
x=[data(:,1) data(:,2) data(:,3) data(:,4)]
covarianceMat=cov(x)
correlationMat=corrcoef(x)
logLikelihood=sum(log(normpdf(a,mu1,sigma1)))+sum(log(normpdf(b,mu2,sigma2)))+sum(log(normpdf(c,mu3,sigma3)))+sum(log(normpdf(d,mu4,sigma4)))
pA=sum(log(normpdf(a,mu1,sigma1)))
tempData=[a d]
tempMu=[mu1 mu4]
tempCo=cov(tempData)
pAjointD=sum(log(mvnpdf(tempData,tempMu,tempCo)))
tempData=[a b d]
tempMu=[mu1 mu2 mu4]
tempCo=cov(tempData)
pAjointBjointD=sum(log(mvnpdf(tempData,tempMu,tempCo)))
tempData=[d a]
tempMu=[mu4 mu1]
tempCo=cov(tempData)
pDjointA=sum(log(mvnpdf(tempData,tempMu,tempCo)))
tempData=[b a d]
tempMu=[mu2 mu1 mu4]
tempCo=cov(tempData)
pBjointAjointD=sum(log(mvnpdf(tempData,tempMu,tempCo)))
pBgivenAD=pBjointAjointD-pAjointD
tempData=[c a b d]
tempMu=[mu3 mu1 mu2 mu4]
tempCo=cov(tempData)
pCjointAjointBjointD=sum(log(mvnpdf(tempData,tempMu,tempCo)))
pCgivenABD=pCjointAjointBjointD-pAjointBjointD
pDgivenA=pDjointA-pA
BNlogLikelihood=pA+pBgivenAD+pCgivenABD+pDgivenA
BNgraph=[0 1 1 1;0 0 1 0;0 0 0 0;0 1 1 0]
view(biograph(BNgraph))
UBitName=['a' 'd' 'h' 'i' 'p' 'v' 'i' 'h']
personNumber=['5' '0' '1' '3' '4' '7' '7' '4']