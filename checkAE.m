function diff = checkAE( images )
%用于检查sparseAutoencoderEpoch函数所得到的梯度grad是否有效
% by 郑煜伟 Aewil 2016-04
% 我们用数值计算梯度的方法得到梯度numGradient（很慢），
% 与sparseAutoencoderEpoch函数（数学分析方法）得到的梯度（很快）进行比较
% 得到两者梯度向量的欧式距离大小（应该非常之小才对）

image = images(:, 1:1);% 因为计算很慢，所以才抽取一个样本（这个图的theta有308308维！）

architecture = [ 784 196 784 ]; % AE网络的结构: inputSize -> hiddenSize -> outputSize
theta = initializeParameters( architecture ); % 依据网络结构初始化网络参数

preOption4AE.isSparse = 1;
option4AE = getAEOption( preOption4AE ); % 得到函数的参数

% 分析方法
[ ~,grad] = calcAEBatch( image, theta, architecture, option4AE );

% 数值计算方法
numGradient = computeNumericalGradient( ...
    @(x) calcAEBatch( image, x, architecture, option4AE ), theta );

% 比较梯度的欧式距离
diff = norm( numGradient - grad ) / norm( numGradient + grad );

end






function numGradient = computeNumericalGradient( fun, theta )
%用数值方法计算 函数fun 在 点theta 处的梯度
% fun：输入类theta，输出实值的函数 y = fun( theta )
% theta：参数向量

    % 初始化 numGradient
    numGradient = zeros( size(theta) );

    % 按微分的原理来计算梯度：变量一个小变化后，函数值得变化程度
    EPSILON   = 1e-4;
    upTheta   = theta;
    downTheta = theta;
    
    wait = waitbar( 0, ['当前进度', num2str(0),'%'] );
    for i = 1: length( theta )
        % waitbar( i/length(theta), wait, ['当前进度', num2str(i/length(theta)),'%'] );
        waitbar( i/length(theta), wait, '当前进度' )
        
        upTheta( i )    = theta( i ) + EPSILON;
        [ resultUp, ~ ] = fun( upTheta );
        
        downTheta( i )    = theta( i ) - EPSILON;
        [ resultDown, ~ ] = fun( downTheta );
        
        numGradient( i )  = ( resultUp - resultDown ) / ( 2 * EPSILON ); % d Vaule / d x
        
        upTheta( i )   = theta( i );
        downTheta( i ) = theta( i );
    end
end
