function predictLabels = predictNN( input, architecture, theta, option )
%计算网络前向阶段，以实现预测
% by 郑煜伟 Aewil 2016-04

startIndex = 1; % 存储变量的下标起点
for i = 1:( length(architecture) - 1 )
    visibleSize = architecture( i );
    hiddenSize  = architecture( i + 1 );
    
    %% 先将 theta 转换为 (W, b) 的矩阵/向量 形式，以便后续处理（与initializeParameters文件相对应）
    endIndex = hiddenSize * visibleSize + startIndex - 1; % 存储变量的下标终点
    W = reshape( theta(startIndex : endIndex), hiddenSize, visibleSize);
    
    if strcmp( option.activation{i}, 'softmax' ) % softmax不需要偏置b
        startIndex = endIndex + 1; % 存储变量的下标起点
    else
        startIndex = endIndex + 1; % 存储变量的下标起点
        endIndex = hiddenSize + startIndex - 1; % 存储变量的下标终点
        b = theta( startIndex : endIndex );
        startIndex = endIndex + 1;
    end
    
    %% feed forward 阶段
    activationFunc = str2func( option.activation{i} ); % 将 激活函数名 转为 激活函数
    % 求隐藏层
    if strcmp( option.activation{i}, 'softmax' ) % softmax不需要偏置b
        hiddenV = W * input; % 求和 -> 诱导局部域V
    else
        hiddenV = bsxfun( @plus, W * input, b ); % 求和 -> 诱导局部域V
    end
    hiddenX = activationFunc( hiddenV ); % 激活函数
    
    clear input
    input = hiddenX;
end

predictLabels = input;

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