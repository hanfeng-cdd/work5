clear;
clc;

german_data=dlmread('german-assignment5.txt');%不足的会自动补0


m=1;     %拉普拉斯平滑参数

test_label=Naive_Bayes(german_data(1:901,:),german_data(902:end,1:end-1),m);


%breast_data=dlmread('breast-cancer-assignment5.txt');%不足的会自动补0

%test_label=Naive_Bayes(breast_data,breast_data(2:end,1:end-1),m);