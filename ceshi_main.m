clear;
clc;

german_data=dlmread('german-assignment5.txt');%����Ļ��Զ���0


m=1;     %������˹ƽ������

test_label=Naive_Bayes(german_data(1:901,:),german_data(902:end,1:end-1),m);


%breast_data=dlmread('breast-cancer-assignment5.txt');%����Ļ��Զ���0

%test_label=Naive_Bayes(breast_data,breast_data(2:end,1:end-1),m);