%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 

MAE = mean(abs(yhat_test - ytest)); % Mean Absolute Error
%disp(MAE)

CS = sum (abs(yhat_test - ytest) <= err_level)/ nTest; % Cummulative Score
disp(CS)
%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides

%--------Code by Nishant Gautam (210832761)---------%
pltcs = zeros(15,1);
for err_level = 1:15
    pltcs(err_level, 1) =  sum (abs(yhat_test - ytest) <= err_level)/ nTest;
end
figure
plot((1:15), pltcs, 'r-.p');
title('Cumulative Score vs. Error Level FG-NET');
xlabel('Error Level (in years)');
ylabel('Cumulative Score');

%--------Code by Nishant Gautam (210832761)---------%
%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.

% ------- code of Nishant Gautam (210832761)-------%
err_level = 5;
% Partial least square regression
[~, ~, ~, ~, BETA] = plsregress(xtrain, ytrain);
yhat_test_pls = [ones(size(xtest,1),1) xtest]*BETA;
MAE_pls = mean(abs(yhat_test_pls - ytest));
CS_pls = sum (abs(yhat_test_pls - ytest) <= err_level)/ nTest;

% the regression tree model
tree = fitrtree(xtrain,ytrain);
yhat_test_tree = predict(tree,xtest);
MAE_tree = mean(abs(yhat_test_tree - ytest));
CS_tree = sum (abs(yhat_test_tree - ytest) <= err_level)/ nTest;
%disp(CS_tree)
% Code by Nishant Gautam (210832761) -----------%

%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox

%-------Code by Nishant Gautam (210832761) ---------%
addpath(genpath('./softwares'));
addpath(genpath('libsvm-3.14'))
svr = fitrsvm(xtrain, ytrain);
yhat_test_svr = predict(svr, xtest);
err_level = 5;
err_svr = abs(yhat_test_svr - ytest);
MAE_svr = sum(err_svr)/size(ytest, 1);
cs_svr = sum (abs(yhat_test_svr - ytest) <= err_level)/nTest;
disp(cs_svr)

fprintf('MAE support vector regression = %f\n',MAE_svr);
fprintf('csNew = %f\n', cs_svr);
%-----------Code by Nishant Gautam (210832761) ---------%



