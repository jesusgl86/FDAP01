function [loss,gradientsSubnet,gradientsParams] = modelLoss(net,fcParams,X1,X2,pairLabels)

% Pass the image pair through the network.
Y = forwardSiamese(net,fcParams,X1,X2);

% Calculate binary cross-entropy loss.
loss = binarycrossentropy(Y,pairLabels);

% Calculate gradients of the loss with respect to the network learnable
% parameters.
[gradientsSubnet,gradientsParams] = dlgradient(loss,net.Learnables,fcParams);

end