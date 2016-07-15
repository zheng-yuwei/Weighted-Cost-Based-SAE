clc, clear;

% 加载数据
[ images4Train, labels4Train ] = loadMNISTData( 'dataSet/train-images.idx3-ubyte',...
    'dataSet/train-labels.idx1-ubyte', 'MinMaxScaler', 0 );

% 检测 计算AE梯度的准确性
[diff, numGradient, grad] = checkAE(images4Train);
fprintf(['AE中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(numGradient - grad)))...
    ' 及 ' num2str(diff) '\n']);

% 检测 计算BP梯度的准确性
[diff, numGradient, grad] = checkBP(images4Train, labels4Train);
fprintf(['AE中计算梯度的分析方法与数值方法的差异性：'...
    num2str(mean(abs(numGradient - grad)))...
    ' 及 ' num2str(diff) '\n']);