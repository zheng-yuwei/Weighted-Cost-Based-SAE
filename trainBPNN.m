function [optTheta, cost] = trainBPNN( input, output, theta, architecture, option4BP )
%训练BP网络
% by 郑煜伟 Aewil 2016-04

% 函数 calcBPBatch 可以根据当前点计算 cost 和 gradient，但是步长不确定
% 这里，调用Mark Schmidt的包来优化迭代 步长：用了l-BFGS
% Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [仅供学术]
addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 100;	  % L-BFGS 的最大迭代代数
options.display = 'off';
% options.TolX = 1e-3;

[optTheta, cost] = minFunc( @(x) calcBPBatch( input, output, x, architecture, option4BP ), ...
    theta, options);


end