function [cost,grad] = calcBPBatch( input, output, theta, architecture, option )
%计算 BPNN 的梯度变化和误差
% by 郑煜伟 Aewil 2016-04
% input：       训练样本集，每一列代表一个样本；
% theta：       权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；
% architecture: 网络结构，每层参数组成的行向量
% 结构体 option
% decayLambda：      权重衰减系数——正则项罚项权重；
% activation：  激活函数类型；

% isBatchNorm：   是否使用 Batch Normalization 来 speed-up学习速度；

% isDenoising：   是否使用 denoising 规则

% 先明确使用BP的规则
% option.isBatchNorm：该规则目前还没加
% option.isDenoising：该规则目前还没加

m                = size( input, 2 ); % 样本数
layers           = length( architecture ); % 网络层数
% 初始化一些参数
layerHiddenV     = cell( 1, layers - 1 ); % 用于盛装每一层神经网络的诱导局部域数据
layerHiddenX     = cell( 1, layers );     % 用于盛装每一层神经网络的输出/输入数据
layerHiddenX{1}  = input;
cost.costRegul   = 0; % 正则项的罚函数
cost.costError   = 0; % cost function
grad             = zeros( size(theta) );
%% feed-forward阶段
startIndex = 1; % 存储变量的下标起点
for i = 1:( layers - 1 )
    visibleSize = architecture( i );
    hiddenSize  = architecture( i + 1 );
    
    activationFunc = str2func( option.activation{i} ); % 将 激活函数名 转为 激活函数
    
    % 先将 theta 转换为 (W, b) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
    endIndex   = hiddenSize * visibleSize + startIndex - 1; % 存储变量的下标终点
    W          = reshape( theta(startIndex : endIndex), hiddenSize, visibleSize);
    
    if strcmp( option.activation{i}, 'softmax' ) % softmax那一层不用偏置b
        startIndex = endIndex + 1; % 存储变量的下标起点
        
        hiddenV = W * input;% 求和 -> 得到诱导局部域 V
    else
        startIndex = endIndex + 1; % 存储变量的下标起点
        endIndex   = hiddenSize + startIndex - 1; % 存储变量的下标终点
        b          = theta( startIndex : endIndex );
        startIndex = endIndex + 1;
        
        hiddenV = bsxfun( @plus, W * input, b ); % 求和 -> 得到诱导局部域 V
    end
    hiddenX = activationFunc( hiddenV ); % 激活函数
    % 计算正则项的罚函数
    cost.costRegul = cost.costRegul + 0.5 * option.decayLambda * sum(sum(W .^ 2));
    
    clear input
    input = hiddenX;
    
    layerHiddenV{ i }     = hiddenV; % 用于盛装每一层神经网络的诱导局部域数据
    layerHiddenX{ i + 1 } = input;   % 用于盛装每一层神经网络的输出/输入数据
end
% 求cost function + regularization
if strcmp( option.activation{layers-1}, 'softmax' ) % 标签类cost
    % softmax的cost，但我并没有求对数，并且加了1. 用于模仿准确率
    indexRow = output';
    indexCol = 1:m;
    index    = (indexCol - 1) .* architecture( end ) + indexRow;
    % cost.costError = sum( 1 - layerHiddenX{layers}(index) ) / m; 
	cost.costError = - sum( log(layerHiddenX{layers}(index)) ) / m; 
else % 实值类cost
    cost.costError = sum( sum((output - layerHiddenX{layers}).^2 ./ 2) ) / m;
end

cost.cost      = cost.costError + cost.costRegul;
cost           = cost.cost;


%% Back Propagation 阶段：链式法则求导
% 求最后一层
activationFuncDeriv = str2func( [option.activation{layers-1}, 'Deriv'] );
if strcmp( option.activation{layers-1}, 'softmax' ) % softmax那一层求导需要额外labels信息
    dError_dOutputV   = activationFuncDeriv( layerHiddenV{layers - 1}, output );
else
    % dError/dOutputV = dError/dOutputX * dOutputX/dOutputV
    dError_dOutputX   = -( output - layerHiddenX{layers} );
    dOutputX_dOutputV = activationFuncDeriv( layerHiddenV{layers - 1} );
    dError_dOutputV   = dError_dOutputX .* dOutputX_dOutputV;
end


% dError/dW = dError/dOutputV * dOutputV/dW
dOutputV_dW = layerHiddenX{ layers - 1 }';
dError_dW   = dError_dOutputV * dOutputV_dW;

if strcmp( option.activation{layers-1}, 'softmax' ) % softmax那一层不用偏置b
    endIndex   = length( theta ); % 存储变量的下标终点
    startIndex = endIndex + 1; % 存储变量的下标起点
else
    % 更新梯度 b
    endIndex   = length( theta ); % 存储变量的下标终点
    startIndex = endIndex - architecture( end )  + 1; % 存储变量的下标起点
    dError_db  = sum( dError_dOutputV, 2 );
    grad( startIndex:endIndex ) = dError_db ./ m;
end
% 更新梯度 W
endIndex   = startIndex - 1; % 存储变量的下标终点
startIndex = endIndex - architecture( end - 1 ) * architecture( end )  + 1; % 存储变量的下标起点
W          = reshape( theta(startIndex:endIndex), architecture( end ), architecture( end - 1 ) );
WGrad      = dError_dW ./ m + option.decayLambda * W;
grad( startIndex:endIndex ) = WGrad(:);

% 误差回传 error back-propagation
for i = ( layers - 2 ):-1:1
    activationFuncDeriv = str2func( [option.activation{i}, 'Deriv'] );
    % dError/dHiddenV = dError/dHiddenX * dHiddenX/dHiddenV
    dError_dHiddenX   = W' * dError_dOutputV; % = dError/dOutputV * dOutputV/dHiddenX
    dHiddenX_dHiddenV = activationFuncDeriv( layerHiddenV{ i } );
    dError_dHiddenV   = dError_dHiddenX .* dHiddenX_dHiddenV;
    % dError/dW1 = dError/dHiddenV * dHiddenV/dW1
    dHiddenV_dW = layerHiddenX{ i }';
    dError_dW   = dError_dHiddenV * dHiddenV_dW;
    
    dError_db = sum( dError_dHiddenV, 2 );
    % 更新梯度 b
    endIndex   = startIndex - 1; % 存储变量的下标终点
    startIndex = endIndex - architecture( i + 1 )  + 1; % 存储变量的下标起点
    % b          = theta( startIndex : endIndex );
    grad( startIndex:endIndex ) = dError_db ./ m;
    
    % 更新梯度 W
    endIndex   = startIndex - 1; % 存储变量的下标终点
    startIndex = endIndex - architecture( i ) * architecture( i + 1 )  + 1; % 存储变量的下标起点
    W          = reshape( theta(startIndex:endIndex), architecture( i + 1 ), architecture( i ) );
    WGrad      = dError_dW ./ m + option.decayLambda * W;
    grad( startIndex:endIndex ) = WGrad(:);
    
    dError_dOutputV = dError_dHiddenV;
end

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
function soft = softmax(x)
    soft = exp(x);
    soft = bsxfun( @rdivide, soft, sum(soft, 1) );
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
function softDeriv = softmaxDeriv( x, labels )
    indexRow = labels';
    indexCol = 1:length(indexRow);
    index    = (indexCol - 1) .* max(labels) + indexRow;
    
%     softDeriv = softmax(x);
%     active   = zeros( size(x) );
%     active(index) = 1;
%     softDeriv = bsxfun( @times, softDeriv - active, softDeriv(index) );

    softDeriv = softmax(x);
    softDeriv(index) = softDeriv(index) - 1;  % 这个是使用原始cost function的导数
end







