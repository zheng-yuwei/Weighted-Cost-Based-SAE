function displayNetwork( weight, figureName, ~ )
%利用网络权重(hiddenSize*inputSize)展示网络所抽取的特征图
% 假设每个 hidden level 1 的 neuron 表示所抽取的一种 feature
% 则连接到 neuron A 的权重向量，代表 input vector 中每一位在 feature A 的重要程度
% 根据权重向量（重要程度），即可构造出 input 的 feature

% 对 每个input位权重 实施归一化
weightMin = min( weight, [], 2 );
weight    = bsxfun( @minus, weight, weightMin );
weightMax = max( weight, [], 2 );
weight    = bsxfun( @rdivide, weight, weightMax );

featureNum  = size( weight, 1 ); % feature数量，也是图片数量
penal       = featureNum * 2 / 3;
picMatCol   = ceil( 1.5 * sqrt(penal) );
picMatRow   = ceil( featureNum / picMatCol );

images = reshape( weight', sqrt( size(weight, 2) ), sqrt( size(weight, 2) ), featureNum ); % 图片
% 展示特征
% 灰度图
if exist( 'figureName', 'var' )
    figure('NumberTitle', 'off', 'Name', figureName );
else
    figure('NumberTitle', 'off', 'Name', 'MNIST手写字体特征图');
end
for i = 1:featureNum
    subplot( picMatRow, picMatCol, i, 'align' );
    imshow( images(:, :, i) );
%     imagesc( images(:, :, i) );
%     axis off;
end

end