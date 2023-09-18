clear all
clc
load('Cos_train_data.mat','Toutput','XTrain','r0','r1')
load('Cos_valid_data.mat','Toutvalid','Xvalid')
load('Cos_test_data.mat','Touttest','Xtest')
%% Training
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
    fullyConnectedLayer(1)
    regressionLayer];

miniBatchSize  = 128;
validationFrequency = floor(numel(Toutput)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','cpu',... %,'auto',... or ,'gpu',... %for GPU
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{Xvalid,Toutvalid}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience', Inf, ...
    'Plots','training-progress', ...
    'Verbose',false);

net=trainNetwork(XTrain,Toutput,layers,options); 
%%
% load CNN_output.mat
YPred1 = predict(net,XTrain);
rmse = sqrt(mean((Toutput - YPred1).^2))
figure
scatter(YPred1,Toutput,'+')
xlabel("Predicted Value",'fontname','timesnewroman')
ylabel("True Value",'fontname','timesnewroman')
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
xlabel("Predicted Value",'fontname','timesnewroman')
ylabel("True Value",'fontname','timesnewroman')
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
xlabel("Predicted Value",'fontname','timesnewroman','Fontsize',24)
ylabel("True Value",'fontname','timesnewroman','Fontsize',24)
ax=gca; ax.FontSize = 16; pbaspect([1 1 1])
hold on
x=r0:.01:r1;
plot(x,x,'color','black','linewidth',2)
R=corrcoef(YPred3,Touttest);R(1,2)
txt = ['r=' num2str(R(1,2))];
textH=text(2,0.5,txt,'FontSize', 18)
