function acc=adboost_Weight(data_type,train_data,test_data,T,m)

[train_num,feature_num]=size(train_data);
%feature_num=feature_num-1;     %去掉标签
[test_num,~]=size(test_data);
test_label=zeros(test_num,1);
D=ones(train_num,1)/train_num;    %初始化分布
%m=0;                              %拉普拉斯平滑系数
%train_label=train_data(:,end);
% label_flag=-1;
%train_ture_label= train_data(:,end);
% if sum(train_ture_label==0)
%     train_ture_label(train_ture_label==0)=-1;
%     label_flag=0;
% end
% train_data(:,end)=train_ture_label;
train_label=train_data(:,end);


w=zeros(T,1);
h={};
%sz = [train_num,1]; % 输出随机矩阵的size
 train_data_D=train_data;
for t=1:T
    t
    %%训练模型得到错误率
    [NaiveBayes_label,h{t}]=Naive_Bayes_Weight(data_type,train_data_D,train_data_D(:,1:end-1),m,D);
    train_label_D=train_data_D(:,end);
    train_label_D(train_label_D==0)=-1;   %把原来为0的转化成-1；
    ErrorRate=sum(D(NaiveBayes_label~=train_label_D));   %基于现在的分布计算准确率
    if ErrorRate>0.5     %出现基分类器错误率大于0.5，则重新采样
        break;
    end
    %计算权值
    w(t)=0.5*log((1-ErrorRate)/ErrorRate);
    %改变数据分布
    label_t=BaseClassifier(data_type,h{t},train_data(:,1:end-1));
    D(label_t==train_label)=D(label_t==train_label)*exp(-w(t));
    D(label_t~=train_label)=D(label_t~=train_label)*exp(w(t));
    D=D/sum(D);   
%    r = discretize(rand(sz),[0 cumsum(D)']);
%    train_data_D=train_data(r,:);
end
%合并基分类器进行预测
if t~=T
    T=t-1;%到底训练出来了多少个分类器,h{t}被丢掉了；
%T=t; 
end
f=zeros(test_num,T);
for t=1:T
    f(:,t)=w(t)*BaseClassifier(data_type,h{t},test_data);
end
test_label=sum(f,2);
test_label(test_label>=0)=1;
test_label(test_label<0)=-1;

acc=sum(test_label==test_data(:,end))/test_num;


