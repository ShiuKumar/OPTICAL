load('sample_data') % load sample data
load('Actual_class') % load the actual class of the sample data
ind = crossvalind('Kfold',Actual_class,10); % generate indices to divide 
% sample data into train and test data

% Select 1st set as test data and remaining sets as test data
num = 1; 

test = (ind == num); 
test_ind = find(test == 1);
train = ~test;
train_ind = find(train == 1);

Train_data = sample_data(:,:,train_ind);
Target_Train = Actual_class(train_ind);
Test_data = sample_data(:,:,test_ind);
Target_Test = Actual_class(test_ind);

% Researchers can provide their own data and parameters to the OPTICAL
% function. Here window size of 50 sample points with 20% overlap is used
predicted_class = OPTICAL(Train_data,Test_data,Target_Train, Target_Test, 50, 20);
