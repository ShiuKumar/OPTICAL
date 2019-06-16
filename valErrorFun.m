function [valError,cons,fileName] = valErrorFun(x, data, Target)
    ind_x = crossvalind('Kfold',Target,5);
    for num = 1:5
        FF_Train = data(ind_x~=num,:);
        Target_Train = Target(ind_x~=num);
        FF_Test = data(ind_x==num,:);
        Target_Test = Target(ind_x==num);

        inputSize = size(FF_Train{1},1)
        numHiddenUnits1 = 100;
        numHiddenUnits2 = 20; 
        maxEpochs = 100;
        MiniBatchSize = 20;

        options = trainingOptions('sgdm', ...
        'ExecutionEnvironment','cpu', ...
        'LearnRateSchedule','piecewise',...
        'InitialLearnRate',x.InitialLearnRate,...
        'L2Regularization',x.L2Regularization,...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',MiniBatchSize, ...
        'GradientThreshold',1, ...
        'Verbose',0);

        numResponses = 1;
        layers = [ ...
        sequenceInputLayer(inputSize)
        lstmLayer(numHiddenUnits1,'OutputMode','sequence')
        lstmLayer(numHiddenUnits2,'OutputMode','last')
        fullyConnectedLayer(numResponses)
        regressionLayer];

        net = trainNetwork(FF_Train,Target_Train',layers,options);
        p11 = predict(net,FF_Train,'MiniBatchSize',MiniBatchSize);
        p1 = double(p11);
        p22 = predict(net,FF_Test,'MiniBatchSize',1);
        p2 = double(p22);
        
        model = svmtrain(Target_Train',p1,['-t 2 -c 0.25 -q']);  
        pred = svmpredict(Target_Test',p2,model);
        disp(size(pred))
        disp(size(Target_Test))
        Error(num) = 1 - mean(pred == Target_Test');
    end
    
    valError = mean(Error);
end
