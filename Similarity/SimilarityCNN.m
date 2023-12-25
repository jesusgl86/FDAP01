clear all
load('Sim_train_data.mat','XTrain1','XTrain2','Toutput')
load('Sim_valid_data.mat','XValid1','XValid2','TValid')
load('Sim_test_data.mat','XTest1','XTest2','TTest')
imageSize = [28 28 1];
N=length(XTrain1);
trainLabels = [repmat(categorical(1), [N, 1]); repmat(categorical(0), [N, 1])];
numWeights=28^2;
layers = [
    imageInputLayer(imageSize,Normalization="none")
    convolution2dLayer(3,8,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")%,'Padding','same')
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    convolution2dLayer(3,32,'Padding','same',WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    reluLayer
    fullyConnectedLayer(numWeights,WeightsInitializer="narrow-normal",BiasInitializer="narrow-normal")
    ];

lgraph = layerGraph(layers);
net = dlnetwork(lgraph);

miniBatchSize = 100;

fcWeights = dlarray(0.01*randn(1,numWeights));
fcBias = dlarray(0.01*randn(1,1));

fcParams = struct(...
    "FcWeights",fcWeights,...
    "FcBias",fcBias);

epochs=100;
learningRate = 1e-3;
gradDecay = 0.99;
gradDecaySq = 0.999;

executionEnvironment = "auto";

figure
C = colororder;
lineLossTrain = animatedline(Color=C(2,:));
lineLossValid = animatedline('LineStyle','--',Color=C(1,:));
ylim([0 inf])
xlabel("Epoch")
ylabel("Loss")
legend("Training","Validation")
grid on

trailingAvgSubnet = [];
trailingAvgSqSubnet = [];
trailingAvgParams = [];
trailingAvgSqParams = [];

start = tic;


M=N/miniBatchSize;
% Loop over mini-batches.
for iteration = 1:epochs
    idx=randperm(N);
    for i=1:M
    Traingid=idx((i-1)*miniBatchSize+1:i*miniBatchSize);
    X1=XTrain1(:,:,:,Traingid);
    X2=XTrain2(:,:,:,Traingid);
    pairLabels=Toutput(Traingid);

    % Convert mini-batch of data to dlarray. Specify the dimension labels
    % "SSCB" is for X dimension, Y dimension, Channel, Batch
    X1 = dlarray(X1,"SSCB");
    X2 = dlarray(X2,"SSCB");

    % Evaluate the model loss and gradients using dlfeval and the modelLoss
    % function listed at the end of the example.
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,X1,X2,pairLabels);

    % Update the Siamese subnetwork parameters.
    [net,trailingAvgSubnet,trailingAvgSqSubnet] = adamupdate(net,gradientsSubnet, ...
        trailingAvgSubnet,trailingAvgSqSubnet,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the fullyconnect parameters.
    [fcParams,trailingAvgParams,trailingAvgSqParams] = adamupdate(fcParams,gradientsParams, ...
        trailingAvgParams,trailingAvgSqParams,iteration,learningRate,gradDecay,gradDecaySq);

    % Update the training loss progress plot.
    D = duration(0,0,toc(start),Format="hh:mm:ss");
    lossValue = double(loss);
    addpoints(lineLossTrain,i/M+(iteration-1),lossValue);
    title("Elapsed: " + string(D))
    drawnow
    end
    Xv1 = dlarray(XValid1,"SSCB");
    Xv2 = dlarray(XValid2,"SSCB");
    pairLabels=TValid;
    [loss,gradientsSubnet,gradientsParams] = dlfeval(@modelLoss,net,fcParams,Xv1,Xv2,pairLabels);
    lossValue = double(loss);
    addpoints(lineLossValid,iteration,lossValue);
end
    Xt1 = dlarray(XTest1,"SSCB");
    Xt2 = dlarray(XTest2,"SSCB");
    pairLabels=TTest;
% Evaluate predictions using trained network
    Y = predictSiamese(net,fcParams,Xt1,Xt2);

    % Convert predictions to binary 0 or 1
    Y = gather(extractdata(Y));
    Y = round(Y);
    NT=length(XTest1);
    accuracy = sum(Y == pairLabels)/(NT)

figure
confusionchart(single(pairLabels),Y)



