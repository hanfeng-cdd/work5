function [test_label,h]=Naive_Bayes2(data_type,train_data,test_data,m)

%data_type=train_data(1,1:end-1);
discrete_num=find(data_type==1);
numerical_num=find(data_type==0);

x=train_data(:,1:end-1);
[train_num,feature_num]=size(x);

x_label=train_data(:,end);   %第一个数据集标签是0和1，第二个是-1和1，下面都用-1和1处理
%label_flag=-1;
if sum(x_label==0)
    x_label(x_label==0)=-1;
%    label_flag=0;
end


p_label=sum(x_label==1);
n_label=sum(x_label==-1);
feature={};

for j=1:length(discrete_num)
    index=discrete_num(j);
    column_x=x(:,index);
    feature{index}.value=unique(column_x);
    value_length=length( feature{index}.value);
    feature{index}.probability_p=zeros(value_length,1);
    feature{index}.probability_n=zeros(value_length,1);
    
    flag=repmat(column_x,[1,value_length])-repmat(feature{index}.value',[train_num,1]);
    flag_1=flag;
    flag_1(x_label==-1,:)=1;
    feature{index}.probability_p=(sum(flag_1==0)+m)/(p_label+value_length*m);
    flag_0=flag;
    flag_0(x_label==1,:)=1;
    feature{index}.probability_n=(sum(flag_0==0)+m)/(n_label+value_length*m);
end

for j=1:length(numerical_num)
    index=numerical_num(j);
    column_x=x(:,index);
  
    feature{index}.pmean = mean(column_x(x_label==1));
    feature{index}.pstd = std(column_x(x_label==1));
    
    feature{index}.nmean=mean(column_x(x_label==-1));
    feature{index}.nstd=std(column_x(x_label==-1));
end

prior_p=(sum(x_label==1)+m)/(train_num+2*m);
prior_n=(sum(x_label==-1)+m)/(train_num+2*m);

h={};                            %分类器
h.feature=feature;
h.prior_p=prior_p;
h.prior_n=prior_n;

%假定测试数据只有特征
test_num=size(test_data,1);
test_label=zeros(test_num,1);
test_p=zeros(test_num,1);
test_n=zeros(test_num,1);
for k=1:test_num
    test_flag=test_data(k,:);    %取出第k个测试例
    
    for j=1:length(discrete_num)  %离散属性,防止数据溢出，用log连加,但不能反复取
            index=discrete_num(j);
            test_p(k)= test_p(k)+log(feature{index}.probability_p(feature{index}.value==test_flag(index)));
            test_n(k)= test_n(k)+log(feature{index}.probability_n(feature{index}.value==test_flag(index)));
    end
    
    
    for j=1:length(numerical_num)
        index=numerical_num(j);
         test_p(k)= test_p(k)+log(normpdf(test_flag(index),feature{index}.pmean,feature{index}.pstd));  %参数是标准差
         test_n(k)= test_n(k)+log(normpdf(test_flag(index),feature{index}.nmean,feature{index}.nstd));   
    end     
end

test_p= test_p+log(prior_p);    %记得加上先验，这是取了对数之后的结果
test_n= test_n+log(prior_n);


test_label(test_p>=test_n)=1;
test_label(test_p<test_n)=-1;   %保持和输入一样



