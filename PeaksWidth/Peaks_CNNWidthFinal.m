clear all
clc
load('Peaks_train_data.mat','ToutputWdt','XTrain')
load('Peaks_valid_data.mat','ToutputWdtvalid','Xvalid')
load('Peaks_test_data.mat','ToutputWdttest','Xtest')
Nm=28;

r0=min(min(Toutput),min(Toutvalid)); r1=max(max(Toutput),max(Toutvalid));
%% Training
layers = [
    imageInputLayer([Nm Nm 1])
%
    convolution2dLayer(3,8)%,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
%
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
    fullyConnectedLayer(1)
    regressionLayer];

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
    'Verbose',false,'ExecutionEnvironment','parallel');

net=trainNetwork(XTrain,Toutput,layers,options); 
%%
% load CNN_output.mat
YPred1 = predict(net,XTrain);
rmse = sqrt(mean((Toutput - YPred1).^2))
figure
scatter(YPred1,Toutput,'+')
xlabel("Predicted Value")
ylabel("True Value")
hold on
x=r0:.01:r1;
plot(x,x,'color','black','linewidth',2)
R=corrcoef(YPred1,Toutput);R(1,2)
txt = ['r=' num2str(R(1,2))];
text(0.2,0.5,txt)
%%
YPred2 = predict(net,Xvalid);
rmse = sqrt(mean((Toutvalid - YPred2).^2))
figure
scatter(YPred2,Toutvalid,'+')
xlabel("Predicted Value")
ylabel("True Value")
hold on
x=r0:.01:r1;
plot(x,x,'color','black','linewidth',2)
R=corrcoef(YPred2,Toutvalid);R(1,2)
txt = ['r=' num2str(R(1,2))];
text(0.2,0.5,txt)
%% Testing
YPred3 = predict(net,Xtest);
rmse = sqrt(mean((Touttest - YPred3).^2))
figure
scatter(YPred3,Touttest,'+')
xlabel("Predicted Value")
ylabel("True Value")
hold on
x=r0:.01:r1;
plot(x,x,'color','black','linewidth',2)
R=corrcoef(YPred3,Touttest);R(1,2)
txt = ['r=' num2str(R(1,2))];
text(0.2,0.5,txt)