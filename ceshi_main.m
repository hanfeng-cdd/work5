clear;
clc;
tic;
german_data=dlmread('german-assignment5.txt');%不足的会自动补0
data_type=german_data(1,1:end-1);
data=german_data(2:end,:);
m=1;     %拉普拉斯平滑参数
T=[1:10];
german_mean_acc=zeros(length(T),1);
german_std_acc=zeros(length(T),1);
for i=1:length(T)
   [german_mean_acc(i),german_std_acc(i)]=Cross_Validation(data_type,data,T(i),m);
   
end
figure
plot(T,german_mean_acc,'*r-')
hold on
plot(T,german_std_acc,'*b-')    


%%
breast_data=dlmread('breast-cancer-assignment5.txt');%不足的会自动补0
data=breast_data(2:end,:);
data_type=breast_data(1,1:end-1);
m=1;     %拉普拉斯平滑参数
T=[1:30];
breast_mean_acc=zeros(length(T),1);
breast_std_acc=zeros(length(T),1);
for i=1:length(T)
   [breast_mean_acc(i),breast_std_acc(i)]=Cross_Validation(data_type,data,T(i),m);
end
figure
plot(T,breast_mean_acc,'*r-')
hold on
plot(T,breast_std_acc,'*b-') 



time=toc