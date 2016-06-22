function theta4SAE = trainSAE( input, output, architecture, option )
%训练Stacked AE
% by 郑煜伟 Aewil 2016-04

option4AE = option.option4AE; % 得到AE的一些预选参数
option4BP = option.option4BP; % 得到BP的一些预选参数

% 设置 weighted cost 权重向量
if option4AE.isWeightedCost
    weightedCost = option4AE.weightedCost;
end

% 初始化网络参数 theta4SAE：用于存储堆叠起来的网络的参数
if strcmp( option4BP.activation, 'softmax' ) % softmax那一层不用偏置b
    countW = architecture * [ architecture(2:end) 0 ]';
    countB = sum( architecture(2:(end - 1)) );
    theta4SAE = zeros( countW + countB, 1 );
else
    countW = architecture * [ architecture(2:end) 0 ]';
    countB = sum( architecture(2:end) );
    theta4SAE = zeros( countW + countB, 1 );
end

%% 多个AE：按 architecture 训练
startIndex = 1; % 存储变量的下标起点
for countAE = 1 : ( length(architecture) - 2 ) % 最后两层用于BP训练
    % AE网络的结构: inputSize -> hiddenSize -> outputSize
    architecture4AE = ...
        [ architecture(countAE) ...
        architecture(countAE + 1) ...
        architecture(countAE) ];
    theta4AE  = initializeParameters( architecture4AE ); % 依据网络结构初始化网络参数
    % 设置 weighted cost 权重向量：根据每层网络结构，修改向量大小
    if option4AE.isWeightedCost
        if countAE == 1
            startWeight = 1;
            endWeight   = architecture( 1 );
            option4AE.weightedCost = weightedCost( startWeight:endWeight );
        else
            startWeight = endWeight + 1;
            endWeight   = endWeight + architecture( countAE );
            option4AE.weightedCost = weightedCost( startWeight:endWeight );
        end
    end
    
    [ optTheta, cost ] = trainAE( input, theta4AE, architecture4AE, countAE, option4AE );
%     if countAE == 1 % 可以根据cost的情况，判断是否还需要继续训练
%         [ optTheta, cost ] = trainAE( input, optTheta, architecture4AE, option4AE );
%     end
    
    disp( ['第' num2str(countAE) '层AE "' ...
        num2str(architecture4AE) '" 的训练误差是：'...
        num2str(cost)] );
    
    % 存储 AE的W1，b1 到 SAE 中
    endIndex = architecture(countAE) * architecture(countAE + 1) + ...
        architecture(countAE + 1) + startIndex - 1;% 存储变量的下标终点
    theta4SAE( startIndex : endIndex ) = optTheta( 1 : ...
        (architecture(countAE) * architecture(countAE + 1) + architecture(countAE + 1)) );
    
    % 修改input为上一层的output
    clear predict theta4AE optTheta cost
    predict = predictNN( input, architecture4AE(1:2),...
        theta4SAE( startIndex : endIndex ), option4AE );
    input = predict;
    
    startIndex = endIndex + 1;
end

%% BP：训练最后两层
architecture4BP = [ architecture(end-1) architecture(end) ]; % 设置 BP 网络结构
% 依据网络结构初始化 BP网络参数
if strcmp( option4BP.activation, 'softmax' ) % softmax那一层不用偏置b
    lastActiveIsSoftmax = 1;
    theta4BP = initializeParameters( architecture4BP, lastActiveIsSoftmax );
else
    theta4BP = initializeParameters( architecture4BP );
end

[ optTheta, cost ] = trainBPNN( input, output, theta4BP, architecture4BP, option4BP ); % 训练BP网络
disp( ['最后一层BP "' num2str(architecture4BP) '" 的训练误差是：' num2str(cost)] );

theta4SAE( startIndex : end ) = optTheta;
    
end