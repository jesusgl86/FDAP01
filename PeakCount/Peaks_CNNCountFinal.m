clear all
clc
load('Peaks_train_data.mat','ToutputPks','XTrain')
load('Peaks_valid_data.mat','ToutputPksvalid','Xvalid')
load('Peaks_test_data.mat','ToutputPkstest','Xtest')
classNames=categories(ToutputPks);
numClasses=numel(classNames);
Nm=28;
%r0=min(min(ToutputPks),min(ToutputPksvalid)); r1=max(max(ToutputPks),max(ToutputPksvalid));
%% 


layers = [
    imageInputLayer([Nm Nm 1])
    convolution2dLayer(5,8)
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(5,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(5,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(5,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    dropoutLayer(0.4)
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer(Name="softmax")
    classificationLayer
    ];

miniBatchSize  = 128;
validationFrequency = floor(numel(ToutputPks)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu',... %,'auto',... or ,'gpu',... %for GPU
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.99, ...
    'LearnRateDropPeriod',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid,ToutputPksvalid}, ...
    'ValidationFrequency',100, ...
    'Plots','training-progress', ...
    'Verbose',false,'ExecutionEnvironment','parallel');

net=trainNetwork(XTrain,ToutputPks,layers,options); 
%%
% load CNN_output.mat
YPred = classify(net,XTrain)';

Trainingaccuracy = sum(YPred == ToutputPks)/numel(ToutputPks)
%%
YPredvalid = classify(net,Xvalid)';

Validationaccuracy = sum(YPredvalid == ToutputPksvalid)/numel(ToutputPksvalid)
%% Testing
YPredtest = classify(net,Xtest)';

Testaccuracy = sum(YPredtest == ToutputPkstest)/numel(ToutputPkstest)