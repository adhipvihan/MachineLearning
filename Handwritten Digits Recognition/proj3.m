% xxxxxxxxxxxxxxxxxxxxx--IMPORTANT--xxxxxxxxxxxxxxxxxxxxx
% This M file needs MNIST Train data with the name 'imageTrain'
% Labels with the name 'labelTrain'


temp=ones(1,length(imageTrain(1,:)));
features=[imageTrain;temp];
phi_total(:,:)=(features(:,:))';
clear temp
%load S:\windows\ml\p3.mat

T=zeros(size(phi_total,1),10);

for i=1:size(phi_total,1)
    j=labelTrain(i,1);
    T(i,j+1)=1;
end
clear i,j

D=785;
K=10;
eta=0.001;
error_initial=inf;
w_old=zeros(D,K);
blr=zeros(1,K);
iteration_count=0;
y=zeros(60000,10);
count=0;



for i=1:60000

	count=count+1

	a=phi_total(i,:)*w_old;
	exp_a=exp(a);
	exp_sum=sum(exp_a);
	
	for j=1:10
		y(i,j)=exp_a(1,j)/exp_sum(1,1);
	end
	delta_E=((phi_total(i,:)'*(y(i,:)-T(i,:))));
	w_old=w_old-(eta.*delta_E);
	error=-sum(sum(T(i,:).*log(y(i,:))));
    
	if error_initial<=error
		iteration_count=0;
		eta=0.001;
    
    else
		iteration_count=iteration_count+1;
    
	if iteration_count>3
       eta=eta+0.005;
       iteration_count=0;
    end
    end
	error_initial=error;
    clear a exp_a exp_sum
end
Wlr=w_old([1:784],:);
blr=ones(1,K);
clearvars -except Wnn1 Wnn2 bnn1 bnn2 UBitName h Wlr blr personNumber err_struct phi_total labelTrain w_old imageTrain




% Code For Neural Network

M=300;
K=10;
D=785;
y=zeros(60000,10);
old_error=inf;
error=inf;
step2=0.0000001;
step1=0.0001;
iteration_count=0;
T=zeros(size(phi_total,1),10);
counter=0;


for i=1:size(phi_total,1)
    j=labelTrain(i,1);
    T(i,j+1)=1;
end
clear i,j

w1=rand(D,M)-0.5;
w2=rand(M,K)-0.5;
count=0;

while((old_error>0.5&&counter<30000))

if(isnan(error))
break
end

count=count+1
    
a_j=phi_total*(w1);

z_j=1./(1+exp(-a_j));

a_k=z_j*(w2);

exp_ak=exp(a_k);

sum_exp_ak=sum(exp_ak,2);

for i=1:60000
    for j=1:10
        y(i,j)=exp_ak(i,j)./sum_exp_ak(i,1);
    end
end

error=-sum(sum(T.*log(y)));
counter=counter+1;
err=zeros(1,5);


if old_error<=error
    iteration_count=0;
    step2=0.0000001;
    step1=0.0001;
else
    iteration_count=iteration_count+1;
    if iteration_count>5
        step2=step2+0.0000005;
        step1=step1+0.0005;
        iteration_count=0;
    end

    w1_final=w1;
    w2_final=w2;
    old_error=error;
    err=[err error];
    
end

delta_k=y-T;
delta_j=(z_j.*(1-z_j)).*(delta_k*w2');

delta_E_2=z_j'*delta_k;
delta_E_1=phi_total'*delta_j;


w2=w2-(step2.*delta_E_2);
w1=w1-(step1.*delta_E_1);


end

Wnn1(:,:)=w1_final([1:784],:);
Wnn2=w2_final(:,:);
bnn1=ones(1,M);
bnn2=ones(1,10);
h='sigmoid';
clearvars -except Wnn1 Wnn2 bnn1 bnn2 UBitName h Wlr blr personNumber err