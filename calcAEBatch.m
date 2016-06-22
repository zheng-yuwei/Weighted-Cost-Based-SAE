function [cost,grad] = calcAEBatch( input, theta, architecture, option4AE, inputCorrupted, ~ )
%计算稀疏自编码器的梯度变化和误差
% by 郑煜伟 Aewil 2016-04
% input：       训练样本集，每一列代表一个样本；
% theta：       权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；
% architecture: 网络结构，每层参数组成的行向量
% 结构体 option4AE
% decayLambda： 权重衰减系数——正则项罚项权重；
% activation：  激活函数类型；

% isBatchNorm： 是否使用 Batch Normalization 来 speed-up学习速度；

% isSparse：    是否使用 sparse hidden level 的规则；
% sparseRho：   稀疏性中rho，一般赋值为 0.01；
% sparseBeta：  稀疏性罚项权重；

% isWeightedCost：	是否对每一位数据的cost进行加权对待
% weightedCost：	加权cost的权重

% inputCorrupted： 使用 denoising 规则 则有该参数输入

% 先明确使用AE的规则
% option4AE.isBatchNorm：该规则目前还没加

visibleSize = architecture(1);
hiddenSize  = architecture(2);
% 先将 theta 转换为 (W1, W2, b1, b2) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
W1 = reshape( theta(1 : (hiddenSize * visibleSize)), ...
    hiddenSize, visibleSize);
b1 = theta( (hiddenSize * visibleSize + 1) : (hiddenSize * visibleSize + hiddenSize) );
W2 = reshape( theta((hiddenSize * visibleSize + hiddenSize + 1) : (2 * hiddenSize * visibleSize + hiddenSize)), ...
    visibleSize, hiddenSize);
b2 = theta( (2 * hiddenSize * visibleSize + hiddenSize + 1) : end );

m = size( input, 2 ); % 样本数

%% feed forward 阶段
activationFunc = str2func( option4AE.activation{:} ); % 将 激活函数名 转为 激活函数
% 求隐藏层
if exist( 'inputCorrupted', 'var')
	hiddenV = bsxfun( @plus, W1 * inputCorrupted, b1 ); % 求和 -> V
else
	hiddenV = bsxfun( @plus, W1 * input, b1 ); % 求和 -> V
end
hiddenX = activationFunc( hiddenV ); % 激活函数

% 计算隐藏层的稀疏罚项
if option4AE.isSparse
    rhohat = sum( hiddenX, 2 ) / m;
    KL     = getKL( option4AE.sparseRho, rhohat );
    costSparse = option4AE.sparseBeta * sum( KL );
else
    costSparse = 0;
end

% 求输出层
outputV = bsxfun( @plus, W2 * hiddenX, b2 ); % 求和 -> V
outputX = activationFunc( outputV );   % 激活函数
  
% 求cost function + regularization
if option4AE.isWeightedCost
    costError = sum( sum(option4AE.weightedCost' * (outputX - input).^2) ) / m / 2;
else
    costError = sum( sum((outputX - input).^2) ) / m / 2;
end
costRegul = 0.5 * option4AE.decayLambda * ( sum(sum(W1 .^ 2)) + sum(sum(W2 .^ 2)) );  

% 求总的cost
cost = costError + costRegul + costSparse;


%% Back Propagation 阶段
activationFuncDeriv = str2func( [option4AE.activation{:}, 'Deriv'] );
% 链式法则求导
% dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
if option4AE.isWeightedCost
    dError_dOutputX   = bsxfun( @times, -( input - outputX ), option4AE.weightedCost );
else
    dError_dOutputX   = -( input - outputX );
end
dOutputX_dOutputV = activationFuncDeriv( outputV );
dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
% dError/dW2 = dError/dOutputV * dOutputV/dW2
dOutputV_dW2 = hiddenX';
dError_dW2   = dError_dOutputV * dOutputV_dW2;

W2Grad       = dError_dW2 ./ m + option4AE.decayLambda * W2;
% dError/dHiddenV = ( dError/dHiddenX + dSparse/dHiddenX ) * dHiddenX/dHiddenV
dError_dHiddenX   = W2' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
dHiddenX_dHiddenV = activationFuncDeriv( hiddenV );
if option4AE.isSparse
    dSparse_dHiddenX = option4AE.sparseBeta .* getKLDeriv( option4AE.sparseRho, rhohat );
    dError_dHiddenV  = (dError_dHiddenX + repmat(dSparse_dHiddenX, 1, m)) .* dHiddenX_dHiddenV;
else
    dError_dHiddenV  = dError_dHiddenX .* dHiddenX_dHiddenV;
end
% dError/dW1 = dError/dHiddenV * dHiddenV/dW1
dHiddenV_dW1 = input';
dError_dW1   = dError_dHiddenV * dHiddenV_dW1;

W1Grad       = dError_dW1 ./ m + option4AE.decayLambda * W1;


% 用于解释梯度消失得厉害！！！
% disp( '梯度消失' );
% disp( [ 'W2梯度绝对值均值：', num2str(mean(mean(abs(W2Grad)))), ...
%     ' -> ','W1梯度绝对值均值：', num2str(mean(mean(abs(W1Grad)))) ] );
% disp( [ 'W2梯度最大值：', num2str(max(mean(W2Grad))), ...
%     ' -> ','W1梯度最大值：', num2str(max(mean(W1Grad))) ] );


% 求偏置的导数
dError_db2 = sum( dError_dOutputV, 2 );
b2Grad     = dError_db2 ./ m;
dError_db1 = sum( dError_dHiddenV, 2 );  
b1Grad     = dError_db1 ./ m;

grad = [ W1Grad(:); b1Grad(:); W2Grad(:); b2Grad(:) ];

end  


%% 激活函数
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));  
end
% tanh有自带函数
function x = reLU(x)
    x(x < 0) = 0;
end
function x = almostReLU(x)
    x(x < 0) = x(x < 0) * 0.2;
end
%% 激活函数导数
function sigmDeriv = sigmoidDeriv(x)
    sigmDeriv = sigmoid(x).*(1-sigmoid(x));  
end
function tanDeriv = tanhDeriv(x)
    tanDeriv = 1 ./ cosh(x).^2; % tanh的导数
end
function x = reLUDeriv(x)
    x(x < 0) = 0;
    x(x > 0) = 1;
end
function x = almostReLUDeriv(x)
    x(x < 0) = 0.2;
    x(x > 0) = 1;
end

%% KL散度函数及导数
function KL = getKL(sparseRho,rhohat)
%KL-散度函数
    EPSILON = 1e-8; %防止除0
    KL = sparseRho .* log( sparseRho ./ (rhohat + EPSILON) ) + ...
        ( 1 - sparseRho ) .* log( (1 - sparseRho) ./ (1 - rhohat + EPSILON) );  
end

function KLDeriv = getKLDeriv(sparseRho,rhohat)
%KL-散度函数的导数
    EPSILON = 1e-8; %防止除0
    KLDeriv = ( -sparseRho ) ./ ( rhohat + EPSILON ) + ...
        ( 1 - sparseRho ) ./ ( 1 - rhohat + EPSILON );  
end