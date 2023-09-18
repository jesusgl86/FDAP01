clear all
clc
load('Conv_train_data.mat','Toutput','XTrain')
load('Conv_valid_data.mat','Toutvalid','Xvalid')
load('Conv_test_data.mat','Touttest','Xtest')
classNames=categories(Toutput);
numClasses=numel(classNames);
Nm=28;
%r0=min(min(ToutputPks),min(ToutputPksvalid)); r1=max(max(ToutputPks),max(ToutputPksvalid));
%% 


layers = [
    imageInputLayer([28 28 1])
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
    classificationLayer
    ];

miniBatchSize  = 128;
validationFrequency = floor(numel(Toutput)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu',... %,'auto',... or ,'gpu',... %for GPU
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.99, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid,Toutvalid}, ...
    'ValidationFrequency',100, ...
    'Plots','training-progress', ...
    'Verbose',false);

net=trainNetwork(XTrain,Toutput,layers,options); 
%%
% load CNN_output.mat
YPred = classify(net,XTrain);

Trainingaccuracy = sum(YPred == Toutput)/numel(Toutput)
%%
YPredvalid = classify(net,Xvalid);

Validationaccuracy = sum(YPredvalid == Toutvalid)/numel(Toutvalid)
%% Testing
YPredtest = classify(net,Xtest);

Testaccuracy = sum(YPredtest == Touttest)/numel(Touttest)