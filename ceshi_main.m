clear;
clc;
tic;
german_data=dlmread('german-assignment5.txt');%不足的会自动补0
data_type=german_data(1,1:end-1);
train_data=german_data(2:end,1:end);
test_data=german_data(2:end,1:end-1);

%data=Watermelon3();
%test=[1 1 1 1 1 1 0.697 0.46];
m=1;     %拉普拉斯平滑参数
T=20;
test_label=adboost(data_type,train_data,test_data,T);
ture_label=german_data(2:end,end);
sum(test_label==ture_label)
%test_label=Naive_Bayes2(data_type,german_data,german_data(2:end,1:end-1),m);
%test_label=Naive_Bayes(data,test,m);


%breast_data=dlmread('breast-cancer-assignment5.txt');%不足的会自动补0

%test_label=Naive_Bayes(breast_data,breast_data(2:end,1:end-1),m);


time=toc