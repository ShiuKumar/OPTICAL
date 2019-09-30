load('sample_data')
load('Actual_class')
ind = crossvalind('Kfold',Actual_class,10);

num = 1;

test = (ind == num); 
test_ind = find(test == 1);
train = ~test;
train_ind = find(train == 1);

Train_data = sample_data(:,:,train_ind);
Target_Train = Actual_class(train_ind);
Test_data = sample_data(:,:,test_ind);
Target_Test = Actual_class(test_ind);

predicted_class = OPTICAL(Train_data,Test_data,Target_Train, Target_Test, 50, 20);