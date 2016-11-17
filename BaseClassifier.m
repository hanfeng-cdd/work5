function label=BaseClassifier(data_type,h,test_data)

discrete_num=find(data_type==1);
numerical_num=find(data_type==0);

test_num=size(test_data,1);
label=zeros(test_num,1);
test_p=zeros(test_num,1);
test_n=zeros(test_num,1);
for k=1:test_num
    test_flag=test_data(k,:);    %取出第k个测试例
    
    for j=1:length(discrete_num)  %离散属性,防止数据溢出，用log连加,但不能反复取
        index=discrete_num(j);
       % h.feature{index}.probability_p(h.feature{index}.value==test_flag(index))         %出现了新的特征
        test_p(k)= test_p(k)+log(h.feature{index}.probability_p(h.feature{index}.value==test_flag(index)));
        test_n(k)= test_n(k)+log(h.feature{index}.probability_n(h.feature{index}.value==test_flag(index)));
    end
    
    for j=1:length(numerical_num)
        index=numerical_num(j);
         test_p(k)= test_p(k)+log(normpdf(test_flag(index),h.feature{index}.pmean,h.feature{index}.pstd));  %参数是标准差
         test_n(k)= test_n(k)+log(normpdf(test_flag(index),h.feature{index}.nmean,h.feature{index}.nstd));   
    end     
end

test_p= test_p+log(h.prior_p);    %记得加上先验，这是取了对数之后的结果
test_n= test_n+log(h.prior_n);


label(test_p>=test_n)=1;
label(test_p<test_n)=-1;   %保持和输入一样





end