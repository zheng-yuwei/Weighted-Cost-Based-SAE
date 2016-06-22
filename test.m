% %测试用的文件
% % by 郑煜伟 Aewil 2016-04
% 
% %% 验证AE梯度计算的正确性
% % diff = checkAE( images ); % 已经验证，除非有时间，别运行！慢！天的那种
% % disp(diff); % diff应该很小
% 
% %% 测试 sparse DAE：训练一层 sparse DAE，将重构数据与原数据进行对比 - DAE通过
% clc;clear
% % 用到 loadMNISTImages，getAEOption，initializeParameters，trainAE函数
% [ input, labels ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
%     'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 1 );
% architecture = [ 784 196 784 ];
% % 设置 AE的预选参数 及 BP的预选参数
% preOption4SAE.option4AE.isSparse    = 1;
% preOption4SAE.option4AE.isDenoising = 1;
% preOption4SAE.option4AE.activation  = { 'reLU' };
% % 得到SAE的预选参数
% option4SAE = getSAEOption( preOption4SAE );
% option4AE = option4SAE.option4AE;
% 
% countAE = 1;
% 
% theta = initializeParameters( architecture );
% [optTheta, cost] = trainAE( input, theta, architecture, countAE, option4AE );
% 
% % 将训练好的AE所重构出来的图片输出，与原始图片进行对比
% option4AE.activation = { 'reLU'; 'reLU' };
% predict = predictNN( input, architecture, optTheta, option4AE );
% 
% imagesPredict = reshape( predict, sqrt(size(predict, 1)), sqrt(size(predict, 1)), size(predict, 2) );
% % 灰度图
% figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片(重构）');
% showImagesNum = 200;
% penal         = showImagesNum * 2 / 3;
% picMatCol     = ceil( 1.5 * sqrt(penal) );
% picMatRow     = ceil( showImagesNum / picMatCol );
% for i = 1:showImagesNum
%     subplot( picMatRow, picMatCol, i, 'align' );
%     imshow( imagesPredict(:, :, i) );
% end
% % 热量图 jet
% figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片(重构）-热量图');
% for i = 1:showImagesNum
%     subplot( picMatRow, picMatCol, i, 'align' );
%     imagesc( imagesPredict(:, :, i) );
%     axis off;
% end
% % 绘制权重向量w来表示特征

%% 计算得到 weight 后，测试加 weight 的准确率
% 读取 image 及 label
[ images4Train0, labels4Train0 ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 0 );
images4Train = images4Train0( :, 1:6000 );
labels4Train = labels4Train0( 1:6000, 1 );
[ images4Test, labels4Test ] = loadMNISTData( 'dataSet/t10k-images.idx3-ubyte',...
    'dataSet/t10k-labels.idx1-ubyte', 'MinMaxScaler', 0 );
% 设置 SAE训练时 参数
architecture = [ 784 400 200 10 ]; % SAE网络的结构
% 设置 AE的预选参数 及 BP的预选参数
preOption4SAE.option4AE.activation     = { 'reLU' };
preOption4SAE.option4AE.isSparse       = 1;
preOption4SAE.option4AE.sparseRho      = 0.01;
preOption4SAE.option4AE.sparseBeta     = 0.3;
preOption4SAE.option4AE.isDenoising    = 0;
preOption4SAE.option4AE.noiseRate      = 0.15;
preOption4SAE.option4AE.isWeightedCost = 1;

preOption4SAE.option4BP.activation  = { 'softmax' };
% 得到SAE的预选参数
option4SAE = getSAEOption( preOption4SAE );
% 设置 SAE预测时 的参数
preOption4BPNN.activation = { 'reLU'; 'reLU'; 'softmax' };
option4BPNN = getBPNNOption( preOption4BPNN );

isDispNetwork = 0; % 不展示网络
isDispInfo    = 0; % 不展示信息
accuracy = zeros( 30, 1 );
for i = 1:30
    [ optTheta, accuracy(i, 1) ] = runSAEOnce( images4Train, labels4Train, ...
        images4Test, labels4Test, ... % 数据
        architecture, ...
        option4SAE, option4BPNN, ...
        isDispNetwork, isDispInfo, bestGlobal );
    
    disp( ['第' num2str(i) '次迭代' ] );
end
meanAccuracy = mean( accuracy );
stdAccuracy  = sqrt( sum((accuracy - meanAccuracy) .^ 2) / (size(accuracy, 1) - 1) );
upBound      = meanAccuracy + 1.96 * stdAccuracy;
lowBound     = meanAccuracy - 1.96 * stdAccuracy;
disp( ['置信度 95% 的情况下，准确率为： ['...
    num2str(lowBound) ',' num2str(upBound) ']'] );

