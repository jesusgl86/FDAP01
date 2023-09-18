clear all
clc
%% Simulate pictures
n=1000;
a=0;
b=5;
vx=linspace(a,b,n+1);
sigma=1;
g=@(vx) exp(vx)+sigma*randn(size(vx));;
r0=0; r1=5;
% Making training data
N=1000;
Toutput=zeros(N,1);
XTrain=zeros(28,28,1,N);
%%
for i=1:N
    i
    % Function
    %r=2*(rand-.5);
    r=2*rand+(a+b)/2;
    y=sign(rand-.5);
    xouttrain0(i,:)=g(y*(vx-r));
    xouttrain(i,:)=xouttrain0(i,:)/(max(xouttrain0(i,:)));
    M=diffmat(xouttrain(i,:));
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
   %I=XTrain(:,:,1,1), figure('visible','on'),imshow(I)
    Toutput(i)=y;
    XTrain(:,:,1,i)=I;
end
Toutput=categorical(Toutput);
figure('visible','on'),imshow(XTrain(:,:,1,1))
Toutput(1)
save('Inc_train_data.mat','Toutput','XTrain','r0','r1')
%% Making validation data
N=100;
Toutvalid=zeros(N,1);
Xvalid=zeros(28,28,1,N);
%%
for i=1:N
    
    i
    % Function
    %r=2*(rand-.5);
    r=2*rand+(a+b)/2;
    y=sign(rand-.5);
    xoutvalid0(i,:)=g(y*(vx-r));
    xoutvalid(i,:)=xoutvalid0(i,:)/(max(xoutvalid0(i,:)));
    M=diffmat(xoutvalid(i,:));
    % plott(g(r));
    % plot
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Toutvalid(i)=y;
    Xvalid(:,:,1,i)=I;
end
Toutvalid=categorical(Toutvalid);

save('Inc_valid_data.mat','Toutvalid','Xvalid')
%% Making testing data
N=100;
Touttest=zeros(N,1);
Xtest=zeros(28,28,1,N);
%%
for i=1:N
    
    i
    % Function
    %r=2*(rand-.5);
    r=2*rand+(a+b)/2;
    y=sign(rand-.5);
    xouttest0(i,:)=g(y*(vx-r));
    xouttest(i,:)=xouttest0(i,:)/(max(xouttest0(i,:)));
    M=diffmat(xouttest(i,:));
    I = mat2gray(M);
    I=imresize(I,[28 28]);
    Touttest(i)=y;
    Xtest(:,:,1,i)=I;
end
Touttest=categorical(Touttest);
save('Inc_test_data.mat','Touttest','Xtest')
