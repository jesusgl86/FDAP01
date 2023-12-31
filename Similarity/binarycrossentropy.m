
function loss = binarycrossentropy(Y,pairLabels)

% Get precision of prediction to prevent errors due to floating point
% precision.
precision = underlyingType(Y);

% Convert values less than floating point precision to eps.
Y(Y < eps(precision)) = eps(precision);

% Convert values between 1-eps and 1 to 1-eps.
Y(Y > 1 - eps(precision)) = 1 - eps(precision);

% Calculate binary cross-entropy loss for each pair
loss = -pairLabels.*log(Y) - (1 - pairLabels).*log(1 - Y);

% Sum over all pairs in minibatch and normalize.
loss = sum(loss)/numel(pairLabels);

end