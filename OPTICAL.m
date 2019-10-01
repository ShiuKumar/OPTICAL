function [predicted_class] = OPTICAL(train_data, ...
                                     test_data, ...
                                     class_train, ...
                                     class_test, ...
                                     window_size, ...
                                     percentage_overlap, ...
                                     varargin)
%%
% The code can be used for research purpose only. Any work resulting from 
% the use of this code should cite our original paper "Brain wave 
% classification using long short-term memory network based OPTICAL 
% predictor" published in Scientific Reports in June, 2019

%**************************************************************************
%  The OPTICAL predictor
%**************************************************************************
%   train_data: consists of the filtered samples (trials) of size ch x t x n
%   where ch is the number of channels data, t is the number of sample
%   points and n is the number of samples (trials)
%   test_data: same size as train_data with different number of trials
%   class_train and class_test contain the actual target class for train
%   and test data
%   window_size: size of the window for applying sliding window to obtain 
%   feature matrix for the LSTM network as mentioned in the paper
%   percentage_overlap: is the amount of overlap of window_size in percentage 
%   optional (varargin): optionally, the initial_learn_rate_range and
%   L2_regularization_range can be specified

if nargin==1
    initial_learn_rate_range = varargin(1);
elseif nargin == 2
    initial_learn_rate_range = varargin(1);
    L2_regularization_range = varargin(2);
else % set default range
    initial_learn_rate_range = [1e-3 1e-1]; 
    L2_regularization_range = [1e-5 1e-3];
end

optimVars = [
optimizableVariable('InitialLearnRate',initial_learn_rate_range,'Transform','log')
optimizableVariable('L2Regularization',L2_regularization_range,'Transform','log')
];    
        
original_window_size = size(train_data,2);
overlap = round(percentage_overlap/100*window_size);
number_of_windows = floor((original_window_size - window_size)/overlap);

for i = 1:length(class_train)
    for j = 1:number_of_windows
        Train_data1(:,:,(i-1)*number_of_windows+j) = train_data(:,(j-1)*overlap+1:(j-1)*overlap+window_size,i);
        Target_Train1((i-1)*number_of_windows+j) = class_train(i);
    end
end

for i = 1:length(class_test)
    for j = 1:number_of_windows
        Test_data1(:,:,(i-1)*number_of_windows+j) = test_data(:,(j-1)*overlap+1:(j-1)*overlap+window_size,i);
        Target_Test1((i-1)*number_of_windows+j) = class_test(i);
    end
end

[Z, Wcsp] = CSP(Train_data1,Target_Train1',3);
[Z1, Wcsp1] = CSP(train_data,class_train',3);

clear F
clear F1

for i = 1:1:size(Z,3)
    var1 = var(Z(:,:,i)');
    F(i,:) = log(var1);%./log(sum(var1));
end

for i = 1:1:size(Z1,3)
    var1 = var(Z1(:,:,i)');
    F1(i,:) = log(var1./sum(var1));
end

clear Z_Test
clear Z1_Test
clear F_Test
clear F1_Test

for i = 1:1:size(Target_Test1,2)
    Z_Test(:,:,i) = Wcsp*Test_data1(:,:,i); % Z - csp transformed data
    var1 = var(Z_Test(:,:,i)');
    F_Test(i,:) = log(var1);   
end

for i = 1:1:length(class_test)
    Z1_Test(:,:,i) = Wcsp1*real(test_data(:,:,i)); % Z - csp transformed data
    var1 = var(Z1_Test(:,:,i)');
    F1_Test(i,:) = log(var1./sum(var1));
end

F(isnan(F)) = 1E-3;
F1(isnan(F1)) = 1E-3;

[y2, Wlda2] = LDA(F1,class_train',2);
y2_Test = Wlda2'*F1_Test';

for i = 1:length(class_train)
    FF_Train{i,1} = F((i-1)*number_of_windows+1:i*number_of_windows,:)';
end

for i = 1:length(class_test)
    FF_Test{i,1} = F_Test((i-1)*number_of_windows+1:i*number_of_windows,:)';
end

fun = @(x)valErrorFun(x, FF_Train, class_train);

results = bayesopt(fun,optimVars,...
'IsObjectiveDeterministic',true,...
'MaxObj',25,...
'MaxTime',5*60,...
'UseParallel',true);
close all

inputSize = size(F,2);
% This following parameters can be changed to your own network size, 
% however, the same needs to be done in valErrorFun as these parameters 
% needs to be same
numHiddenUnits1 = 100; 
numHiddenUnits2 = 20; 
maxEpochs = 100;
MiniBatchSize = 20;
        
numResponses = 1;
layers = [ ...
sequenceInputLayer(inputSize)
lstmLayer(numHiddenUnits1,'OutputMode','sequence')
lstmLayer(numHiddenUnits2,'OutputMode','last')
fullyConnectedLayer(numResponses)
regressionLayer];

options = trainingOptions('sgdm', ...
'ExecutionEnvironment','cpu', ...
'LearnRateSchedule','piecewise',...
'InitialLearnRate',results.XAtMinObjective.InitialLearnRate ,...
'L2Regularization',results.XAtMinObjective.L2Regularization,...
'MaxEpochs',maxEpochs, ...
'MiniBatchSize',MiniBatchSize, ...
'GradientThreshold',1, ...
'Verbose',0);

net = trainNetwork(FF_Train,class_train',layers,options);

p1 = predict(net,FF_Train,'MiniBatchSize',1);
MODEL=fitcsvm([p1 y2],class_train','Solver','L1QP');
predicted_class_train = predict(MODEL,[p1 y2]);

p2 = predict(net,FF_Test,'MiniBatchSize',1);
predicted_class = predict(MODEL,[p2 y2_Test']);

train_accuracy = mean(class_train == predicted_class_train')*100;
test_accuracy = mean(class_test == predicted_class')*100;
fprintf('Accuracy on train data is %5.2f %',train_accuracy)
fprintf('\nAccuracy on test data is %5.2f %',test_accuracy)
fprintf('\n')
