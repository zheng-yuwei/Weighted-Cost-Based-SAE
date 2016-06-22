% by 郑煜伟 Aewil 2016-04
clc;clear
%% 读取 image 及 label
[ images4Train0, labels4Train0 ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 0 );
images4Train = images4Train0( :, 1:6000 );
labels4Train = labels4Train0( 1:6000, 1 );
[ images4Test, labels4Test ] = loadMNISTData( 'dataSet/t10k-images.idx3-ubyte',...
    'dataSet/t10k-labels.idx1-ubyte', 'MinMaxScaler', 0 );
	
%% 设置 SAE训练时 参数
architecture = [ 784 400 200 10 ]; % SAE网络的结构
% 设置 AE的预选参数 及 BP的预选参数
preOption4SAE.option4AE.activation     = { 'reLU' };
preOption4SAE.option4AE.isSparse       = 1;
preOption4SAE.option4AE.sparseRho      = 0.01;
preOption4SAE.option4AE.sparseBeta     = 0.3;
preOption4SAE.option4AE.isDenoising    = 1;
preOption4SAE.option4AE.noiseRate      = 0.15;
preOption4SAE.option4AE.isWeightedCost = 1;

preOption4SAE.option4BP.activation  = { 'softmax' };
% 得到SAE的预选参数
option4SAE = getSAEOption( preOption4SAE );
%% 设置 SAE预测时 的参数
preOption4BPNN.activation = { 'reLU'; 'reLU'; 'softmax' };
option4BPNN = getBPNNOption( preOption4BPNN );


%% 求解SAE网络
if option4SAE.option4AE.isWeightedCost % 用PSO优化 Weighted Cost
    isDispNetwork = 0; % 不展示网络
    isDispInfo    = 0; % 不展示信息
    fun = @(x) runSAEOnce( images4Train, labels4Train, ...
        images4Test, labels4Test, ... % 数据
        architecture, ...
        option4SAE, option4BPNN, ...
        isDispNetwork, isDispInfo, x );
    % 设置PSO参数：种群大小 和 迭代次数
    option4PSO.population = 1;
    option4PSO.iteration  = 1;
    % 开始使用PSO优化SAE网络
    [ optTheta, bestGlobal, bestGlobalFit ] = optWeightedCostByPSO( fun, architecture, option4PSO );
	
	disp( ['MNIST测试集 SAE(微调 + weighted Cost）准确率为： ', num2str(bestGlobalFit * 100), '%'] );
    % 更新求解出来的 Weighted Cost
    option4SAE.option4AE.weightedCost = bestGlobal;
else % 直接求解SAE网络
	%% 运行SAE一次
    isDispNetwork = 0; % 不展示网络
    isDispInfo    = 1; % 展示信息
	[ optTheta, accuracy ] = runSAEOnce( images4Train, labels4Train, ...
		images4Test, labels4Test, ... % 数据
		architecture, ...
		option4SAE, option4BPNN, ...
		isDispNetwork, isDispInfo );
	
    % 因为设置了 isDispInfo = 1，所以就不用再展示了
% 	disp( ['MNIST测试集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%'] );
end

%% 用SAE进行预测 —— 因为 runSAEOnce中已经预测了一次，所以这里注释掉
% predictLabels = predictNN( images4Test, architecture, optTheta, option4BPNN );
% accuracy = getAccuracyRate( predictLabels, labels4Test );

if exist( 'bestGlobalFit', 'var' )
    save result optTheta bestGlobalFit bestGlobal
    mail2Me( 'Finished', ['结果为：' num2str(bestGlobalFit)], 'result.mat' )
else
    save result optTheta accuracy
    mail2Me( 'Finished', ['结果为：' num2str(accuracy)], 'result.mat' );
end

