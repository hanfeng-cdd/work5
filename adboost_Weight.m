function test_label=adboost_Weight(data_type,train_data,test_data,T,m)

[train_num,feature_num]=size(train_data);
feature_num=feature_num-1;     %ȥ����ǩ
[test_num,~]=size(test_data);
test_label=zeros(test_num,1);
D=ones(train_num,1)/train_num;    %��ʼ���ֲ�
%m=0;                              %������˹ƽ��ϵ��
%train_label=train_data(:,end);
label_flag=-1;
train_ture_label= train_data(:,end);
if sum(train_ture_label==0)
    train_ture_label(train_ture_label==0)=-1;
    label_flag=0;
end
train_data(:,end)=train_ture_label;



w=zeros(T,1);
h={};
%sz = [train_num,1]; % �����������size
 train_data_D=train_data;
for t=1:T
    %%ѵ��ģ�͵õ�������
    [NaiveBayes_label,h{t}]=Naive_Bayes_Weight(data_type,train_data_D,train_data_D(:,1:end-1),m,D);
    train_label_D=train_data_D(:,end);
    train_label_D(train_label_D==0)=-1;   %��ԭ��Ϊ0��ת����-1��
    ErrorRate=sum(D(NaiveBayes_label~=train_label_D));   %�������ڵķֲ�����׼ȷ��
    if ErrorRate>0.5     %���ֻ������������ʴ���0.5�������²���
        break;
    end
    %����Ȩֵ
    w(t)=0.5*log((1-ErrorRate)/ErrorRate);
    %�ı����ݷֲ�
    D(NaiveBayes_label==train_label_D)=D(NaiveBayes_label==train_label_D)*exp(-w(t));
    D(NaiveBayes_label~=train_label_D)=D(NaiveBayes_label~=train_label_D)*exp(w(t));
    D=D/sum(D);   
%    r = discretize(rand(sz),[0 cumsum(D)']);
%    train_data_D=train_data(r,:);
end
%�ϲ�������������Ԥ��
if t~=T
    T=t-1;%����ѵ�������˶��ٸ�������,h{t}�������ˣ�
%T=t;   
end
f=zeros(test_num,T);
for t=1:T
    f(:,t)=w(t)*BaseClassifier(data_type,h{t},test_data);
end
test_label=sum(f,2);
test_label(test_label>=0)=1;
test_label(test_label<0)=label_flag;


