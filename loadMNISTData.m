function [ images, labels ] = loadMNISTData( imagesFile, labelsFile, preProcess, isShowImages, varargin )
%加载MNIST数据集：images和labels
% by 郑煜伟 Aewil 2016-04

if exist( 'isShowImages', 'var' )
    images = loadMNISTImages(  imagesFile, preProcess, isShowImages );
else
    images = loadMNISTImages(  imagesFile );
end
labels = loadMNISTLabels( labelsFile );

end

function images = loadMNISTImages(  fileName, preProcess, isShowImages, varargin )
%返回一个  #像素点数 * #样本数 的矩阵

    %% 读取 raw MNIST images
    fp = fopen( fileName, 'rb' );
    assert( fp ~= -1, [ 'Could not open ', fileName, ' ' ] );  % 打不开则报错

    magic = fread( fp, 1, 'int32', 0, 'ieee-be' );
    assert( magic == 2051, [ 'Bad magic number in ', fileName, ' ' ] ); % 规定的 magic number，用于check文件是否正确

    numImages = fread( fp, 1, 'int32', 0, 'ieee-be' ); % 连续读出三个关于文件数据属性的数
    numRows   = fread( fp, 1, 'int32', 0, 'ieee-be' );
    numCols   = fread( fp, 1, 'int32', 0, 'ieee-be' );

    images = fread( fp, inf, 'unsigned char' );
    images = reshape( images, numCols, numRows, numImages ); % 文件数据是按行排列的，而matlab是按列排列的。
    images = permute( images, [ 2 1 3 ] );

    fclose( fp );
    %% 显示200张images
    if exist( 'isShowImages', 'var' ) &&  isShowImages == 1
        figure('NumberTitle', 'off', 'Name', 'MNIST手写字体图片');
        showImagesNum = 200;
        penal         = showImagesNum * 2 / 3;
        picMatCol     = ceil( 1.5 * sqrt(penal) );
        picMatRow     = ceil( showImagesNum / picMatCol );
        for i = 1:showImagesNum
            subplot( picMatRow, picMatCol, i, 'align' );
            imshow( images(:, :, i) );
        end
    end

    %% 对 images 进行处理
    % 转化为 #像素点数 * #样本数 矩阵
    images = reshape( images, size(images, 1) * size(images, 2), size(images, 3) );
    
    if strcmp( preProcess, 'MinMaxScaler' )
        % 归一化到 [0,1]
        images = double( images ) / 255; % 激活函数值域非负
    elseif strcmp( preProcess, 'ZScore' )
        % 标准化处理
        images = zScore( images );% 激活函数值域可正可负
    elseif strcmp( preProcess, 'Whitening' )
        % 白化
        images = whitening( images ); % 激活函数值域可正可负
    end
end

function data = zScore( data )
%对数据进行标准化处理（样本按列排列）
% 去均值，然后方差缩放
    epsilon = 1e-8; % 防止除0
    data = bsxfun( @minus, data, mean(data, 1) ); % 去均值（这里类似去除图片亮度）
    data = bsxfun( @rdivide, data, sqrt(mean(data .^ 2, 1)) + epsilon ); % 去方差
end
function data = whitening( data )
%对数据进行白化处理（样本按列排列）
% 去均值，然后去相关性
    data = bsxfun( @minus, data, mean(data, 1) ); % 去均值
    [ u, s, ~ ] = svd( data * data' / size(data, 2) ) ; % 求协方差矩阵的svd分解
    data = sqrt(s) \ u' * data; % 白化（去相关性，协方差为1）
end

function labels = loadMNISTLabels( fileName )
%返回一个 #标签数 * #1 的列向量

    %% 读取 raw MNIST labels
    fp = fopen( fileName, 'rb' );
    assert( fp ~= -1, [ 'Could not open ', fileName, ' ' ] );

    magic = fread( fp, 1, 'int32', 0, 'ieee-be' );
    assert( magic == 2049, [ 'Bad magic number in ', fileName, ' ' ] );

    numLabels = fread( fp, 1, 'int32', 0, 'ieee-be' );
    labels = fread( fp, inf, 'unsigned char' );

    assert( size(labels, 1) == numLabels, 'Mismatch in label count' );
    fclose( fp );

    labels( labels == 0 ) = 10;

    % 下面本想化成矩阵形式的，后面用softmax就没化了
    % indexRow      = labels';
    % indexCol      = 1:numLabels;
    % index         = (indexCol - 1) .* 10 + indexRow;
    % labels        = zeros( 10, numLabels );
    % labels(index) = 1;
end



