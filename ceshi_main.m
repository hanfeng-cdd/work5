clear;
clc;
tic;
german_data=dlmread('german-assignment5.txt');%不足的会自动补0
data_type=german_data(1,1:end-1);
train_data=german_data(2:end,1:end);
test_data=german_data(2:end,1:end-1);
m=1;     %拉普拉斯平滑参数
T=[1:10:50];
ture_label=german_data(2:end,end);
test_label=zeros(length(ture_label),length(T));
num=zeros(length(T),1);
for i=1:length(T)
    %test_label(:,i)=adboost_Weight(data_type,train_data,test_data,T(i),m);
    test_label(:,i)=adboost(data_type,train_data,test_data,T(i),m);
    num(i)=sum( test_label(:,i)==ture_label);
end
figure
plot(T,num,'*r-')
    


%%
% breast_data=dlmread('breast-cancer-assignment5.txt');%不足的会自动补0
% data_type=breast_data(1,1:end-1);
% train_data=breast_data(2:end,1:end);
% test_data=breast_data(2:end,1:end-1);
% m=1;     %拉普拉斯平滑参数
% T=5;
% test_label=adboost_Weight(data_type,train_data,test_data,T,m);
% ture_label=breast_data(2:end,end);
% sum(test_label==ture_label)
%data=Watermelon3();
%test=[1 1 1 1 1 1 0.697 0.46];

%test_label=adboost(data_type,train_data,test_data,T);

%test_label=Naive_Bayes2(data_type,german_data,german_data(2:end,1:end-1),m);
%test_label=Naive_Bayes(data,test,m);




%test_label=Naive_Bayes(breast_data,breast_data(2:end,1:end-1),m);


time=toc