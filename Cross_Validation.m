function [mean_acc, std_acc]=Cross_Validation(data_type,data,T,m)
%划分交叉验证集
[data_row,~]=size(data);
%都化成-1和1
label=data(:,end);
label(label==0)=-1;
data(:,end)=label;




k=10;
positive_index=find(data(:,end)==1);
positive_index=positive_index(randperm(numel(positive_index)));   %将所有的正的角标随机打乱
negitive_index=find(data(:,end)==-1);
negitive_index=negitive_index(randperm(numel(negitive_index)));
p_num=floor(length(positive_index)/k);
n_num=floor(length(negitive_index)/k);
%p_index=[];
%n_index=[];
index={};
for i=1:k-1
    p_index=positive_index(p_num*(i-1)+1:p_num*i);
    n_index=negitive_index(n_num*(i-1)+1:n_num*i);
    index{i}=[p_index', n_index'];
end
p_index=positive_index(p_num*(k-1)+1:end);
n_index=negitive_index(n_num*(k-1)+1:end);
index{k}=[p_index', n_index'];

acc=zeros(1,k);
totle=1:data_row;
for i=1:k
    train_index=totle;
    train_index(index{i})=[];
   acc(i)=adboost(data_type,data(train_index,:),data(index{i},:),T,m);
%    acc(i)=adboost_Weight(data_type,data(train_index,:),data(index{i},:),T,m);
end
mean_acc=mean(acc);
std_acc=std(acc);


