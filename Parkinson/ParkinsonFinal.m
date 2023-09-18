N=77;
for i = 1:15 
   if i<10
   filename(i,:) = ['C_000',num2str(i) '.txt'];
   else
   filename(i,:) = ['C_00',num2str(i) '.txt'];
   end
end  
for i = 1:62 
   if i<10
   filename(i+15,:) = ['P_000',num2str(i) '.txt'];
   else
   filename(i+15,:) = ['P_00',num2str(i) '.txt'];
   end
end
K=0
Aug1=20
Aug2=5
for k=1:77
    if k~=54
        K=K+1;
        M0 = readmatrix(filename(k,:));
        M0=M0(M0(:,7)==0,:);
        if k<=15
            Aug=Aug1;
        else
            Aug=Aug2;
        end
        for i=1:Aug
            if i==1
                M(:,1)=M0(:,1);
                M(:,2)=M0(:,2);
                M(:,3)=M0(:,3);
                M(:,4)=M0(:,4);
                M(:,5)=M0(:,5);
            else
                M(:,1)=M0(:,1)-20+40*rand;
                M(:,2)=M0(:,2)-50+100*rand;
                M(:,3)=M0(:,3)-.5+rand;
                M(:,4)=M0(:,4)+50*rand;
                M(:,5)=M0(:,5)+20*rand;
            end
            Mx=diffmat(M(:,1));
            Ix = mat2gray(Mx);
            Ix=imresize(Ix,[28 28]);
            My=diffmat(M(:,2));
            Iy = mat2gray(My);
            Iy=imresize(Iy,[28 28]);
            Mz=diffmat(M(:,3));
            Iz = mat2gray(M(:,3));
            Iz=imresize(Iz,[28 28]);
            Mp=diffmat(M(:,4));
            Ip = mat2gray(M(:,1));
            Ip=imresize(Ip,[28 28]);
            Ma=diffmat(M(:,5));
            Ia = mat2gray(M(:,5));
            Ia=imresize(Ia,[28 28]);
            if k<=15
                i+Aug*(k-1)
                XTrain00(:,:,1,i+Aug*(K-1)) =Ix;
                XTrain00(:,:,2,i+Aug*(K-1)) =Iy;
                XTrain00(:,:,3,i+Aug*(K-1)) =Iz;
                XTrain00(:,:,4,i+Aug*(K-1)) =Ip;
                XTrain00(:,:,5,i+Aug*(K-1)) =Ia;
            else
                i+15*Aug1+Aug*(K-15)
                XTrain00(:,:,1,i+15*Aug1+Aug*(K-16)) =Ix;
                XTrain00(:,:,2,i+15*Aug1+Aug*(K-16)) =Iy;
                XTrain00(:,:,3,i+15*Aug1+Aug*(K-16)) =Iz;
                XTrain00(:,:,4,i+15*Aug1+Aug*(K-16)) =Ip;
                XTrain00(:,:,5,i+15*Aug1+Aug*(K-16)) =Ia;
            end
            clear M
        end
    end
    K
end
XTrain0(:,:,1,:)=XTrain00(:,:,1,:);
XTrain0(:,:,2,:)=XTrain00(:,:,2,:);
 n=2

%XYA
% XTrain0(:,:,3,:)=XTrain00(:,:,5,:);
% n=3

%XYP
% XTrain0(:,:,3,:)=XTrain00(:,:,4,:);
% n=3

%XYZ
% XTrain0(:,:,3,:)=XTrain00(:,:,3,:);
% n=3

%XYPA
% XTrain0(:,:,3,:)=XTrain00(:,:,4,:);
% XTrain0(:,:,4,:)=XTrain00(:,:,5,:);
% n=4

%XYZA
% XTrain0(:,:,3,:)=XTrain00(:,:,3,:);
% XTrain0(:,:,4,:)=XTrain00(:,:,5,:);
% n=4

%XYZP
% XTrain0(:,:,3,:)=XTrain00(:,:,3,:);
% XTrain0(:,:,4,:)=XTrain00(:,:,4,:);
% n=4

%XYZPA
% XTrain0(:,:,3,:)=XTrain00(:,:,3,:);
% XTrain0(:,:,4,:)=XTrain00(:,:,4,:);
% XTrain0(:,:,5,:)=XTrain00(:,:,5,:);
% n=5


y(1:15*Aug1,1)=0;
y(1+15*Aug1:15*Aug1+61*Aug2,1)=1;
N=76
numClasses=numel(categories(categorical(y)));
%numClassesTest=0;
%Nt=15;
N=15*Aug1+61*Aug2;
%Ntrain=400
idx=1:N;
%controltrain=0

Xsplit=round(N*.8);
Numvalid=100
%validation set size 20-30, 40-15, 60-10, 120-5
idx=randperm(N,Xsplit);
valididx0=randperm(numel(idx),Numvalid); %
Trainidx=idx(~ismember(1:Xsplit,valididx0));
Valididx=idx(ismember(1:Xsplit,valididx0));
Trainid=ismember(1:N,Trainidx);
Validid=ismember(1:N,Valididx);
Testid=~ismember(1:N,idx);
Xtrain=XTrain0(:,:,:,Trainid);
Xvalid=XTrain0(:,:,:,Validid);
Xtest=XTrain0(:,:,:,Testid);
Toutput=y(Trainid);
Toutvalid=y(Validid);
Touttest=y(Testid);

%%
Toutput=categorical(Toutput);
Toutvalid=categorical(Toutvalid);
Touttest=categorical(Touttest);
classNames=categories(Toutput);
numClasses=numel(classNames);
% imageAugmenter = imageDataAugmenter( 'RandXReflection',true,'RandYReflection',true,'RandXTranslation',[-25 25],'RandYTranslation',[-50 50])
imageSize=[28 28 n]
% augimds = augmentedImageDatastore(imageSize,XTrain,Toutput,'DataAugmentation',imageAugmenter);







%% Training
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3,8)%,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer(Name="softmax")
    classificationLayer];

miniBatchSize  = 20;
%validationFrequency = floor(numel(Toutput)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu',... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',.999, ...
    'LearnRateDropPeriod',50, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid,Toutvalid}, ...
    'ValidationFrequency',20, ...
    'Plots','training-progress', ...
    'Verbose',false);

net1=trainNetwork(Xtrain,Toutput,layers,options); 

%% Testing
YPredtest = classify(net1,Xtest);

Testaccuracy = sum(YPredtest == Touttest)/numel(Touttest)

confusionchart(Touttest,YPredtest)
