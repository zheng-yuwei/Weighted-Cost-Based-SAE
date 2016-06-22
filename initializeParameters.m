function theta = initializeParameters( architecture, lastActiveIsSoftmax, varargin )
%基于每一层神经元数量，随机初始化网络中权重参数
% by 郑煜伟 Aewil 2016-04
% architecture: 网络结构；
% theta：权值列向量，[ W1(:); b1(:); W2(:); b2(:); ... ]；

% 没有传入 lastActiveIsSoftmax，默认不是 softmax激活函数
if nargin == 1
    lastActiveIsSoftmax = 0;
end
% 计算参数个数：W个数，b个数；并初始化。
if lastActiveIsSoftmax % softmax那一层不用偏置b
    countW = architecture * [ architecture(2:end) 0 ]';
    countB = sum( architecture(2:(end-1)) );
    theta = zeros( countW + countB, 1 );
else
    countW = architecture * [ architecture(2:end) 0 ]';
    countB = sum( architecture(2:end) );
    theta = zeros( countW + countB, 1 );
end

% 根据 Hugo Larochelle建议 初始化每层网络的 W
startIndex = 1; % 设置每层网络w的下标起点
for layer = 2:length( architecture )
    % 设置每层网络W的下标终点
    endIndex = startIndex + ...
        architecture(layer)*architecture(layer -1) - 1;
    
    % 权重初始化范围：Hugo Larochelle建议
    r = sqrt( 6 ) / sqrt( architecture(layer) + architecture(layer -1) );  
    
    % (layer -1)  -> layer, f( Wx + b )
    theta(startIndex:endIndex) = rand( architecture(layer) * architecture(layer -1), 1 ) * 2 * r - r;
    
    % 设置下一层网络W的下标起点（跳过b）
    startIndex = endIndex + architecture(layer) + 1;
end

end