function option4SAE = getSAEOption( preOption4SAE, varargin )
%设置SAE的参数
% by 郑煜伟 Aewil 2016-04
% 输入：SAE网络的选项 preOption4SAE
% 返回：SAE网络的选项 option4SAE

if exist( 'preOption4SAE', 'var' )
    % 得到AE的一些预选参数
    if isfield( preOption4SAE, 'option4AE' )
        option4SAE.option4AE = getAEOption( preOption4SAE.option4AE ); 
    else
        option4SAE.option4AE = getAEOption( [] );
    end
    % 得到BP的一些预选参数
    if isfield( preOption4SAE, 'option4BP' )
        option4SAE.option4BP = getBPOption( preOption4SAE.option4BP );
    else
        option4SAE.option4BP = getBPOption( [] );
    end
else
    option4SAE.option4AE = getAEOption( [] ); % 得到AE的一些预选参数
    option4SAE.option4BP = getBPOption( [] ); % 得到BP的一些预选参数
end

end


function option4AE = getAEOption( preOption4AE )
%设置AE的参数
% 输入 AE网络的选项 preOption4AE
% 返回：
% AE网络的选项：option4AE
% decayLambda：		权重衰减系数——正则项罚项权重；
% activation：		激活函数类型：sigmoid，reLU，weaklyReLU，tanh激活函数类型：sigmoid，reLU，weaklyReLU，tanh
% slope：			激活函数为weaklyReLU时，负方向的斜率，默认0.2；

% isBatchNorm：		是否使用 Batch Normalization 来 speed-up学习速度；

% isSparse：		是否使用 sparse hidden level 的规则；
% sparseRho：		稀疏性中rho；
% sparseBeta：		稀疏性罚项权重；

% isDenoising：		是否使用 denoising 规则
% noiseLayer：		AE中添加噪声的层：'firstLayer' or 'allLayers'
% noiseRate：		每一位添加噪声的概率
% noiseMode：		添加噪声的模式：'OnOff' or 'Guass'
% noiseMean：		高斯模式：均值
% noiseSigma：		高斯模式：标准差

% isWeightedCost：	是否对每一位数据的cost进行加权对待
% weightedCost：	加权cost的权重

    if isfield( preOption4AE, 'decayLambda' )
        option4AE.decayLambda = preOption4AE.decayLambda;
    else
        option4AE.decayLambda = 0.01;
    end
    if isfield( preOption4AE, 'activation' )
        option4AE.activation = preOption4AE.activation;
		if strcmp( option4AE.activation{:}, 'weaklyReLU' )
			if isfield( preOption4AE, 'slope' )
				option4AE.slope = preOption4AE.slope;
			else
				option4AE.slope = 0.2;
			end
		end
    else
        option4AE.activation = { 'sigmoid' };
    end

    % batchNorm
    if isfield( preOption4AE, 'isBatchNorm' )
        option4AE.isBatchNorm = preOption4AE.isBatchNorm;
    else
        option4AE.isBatchNorm = 0;
    end

    % sparse
    if isfield( preOption4AE, 'isSparse' )
        option4AE.isSparse = preOption4AE.isSparse;
    else
        option4AE.isSparse = 0;
    end
    if option4AE.isSparse
        if isfield( preOption4AE, 'sparseRho' )
            option4AE.sparseRho = preOption4AE.sparseRho;
        else
            option4AE.sparseRho = 0.1;
        end
        if isfield( preOption4AE, 'sparseBeta' )
            option4AE.sparseBeta = preOption4AE.sparseBeta;
        else
            option4AE.sparseBeta = 0.3;
        end
    end

    % denoising
    if isfield( preOption4AE, 'isDenoising' )
        option4AE.isDenoising = preOption4AE.isDenoising;
		if option4AE.isDenoising
			% denoising每一层 或 只第一个输入层
			if isfield( preOption4AE, 'noiseLayer' )
				option4AE.noiseLayer = preOption4AE.noiseLayer;
			else
				option4AE.noiseLayer = 'firstLayer';
			end
			% 噪声概率
			if isfield( preOption4AE, 'noiseRate' )
				option4AE.noiseRate = preOption4AE.noiseRate;
			else
				option4AE.noiseRate = 0.1;
			end
			% 噪声模式：高斯 或 开关
			if isfield( preOption4AE, 'noiseMode' )
				option4AE.noiseMode = preOption4AE.noiseMode;
			else
				option4AE.noiseMode = 'OnOff';
			end
			switch option4AE.noiseMode
				case 'Guass'
					if isfield( preOption4AE, 'noiseMean' )
						option4AE.noiseMean = preOption4AE.noiseMean;
					else
						option4AE.noiseMean = 0;
					end
					if isfield( preOption4AE, 'noiseSigma' )
						option4AE.noiseSigma = preOption4AE.noiseSigma;
					else
						option4AE.noiseSigma = 0.01;
					end
			end
		end
    else
        option4AE.isDenoising = 0;
    end

    % weightedCost
    if isfield( preOption4AE, 'isWeightedCost' )
        option4AE.isWeightedCost = preOption4AE.isWeightedCost;
    else
        option4AE.isWeightedCost = 0;
    end
    if option4AE.isWeightedCost
        if isfield( preOption4AE, 'weightedCost' )
            option4AE.weightedCost = preOption4AE.weightedCost;
%         else
%             error( '加权cost一定要自己设置权重向量！' );
        end
    end
end


function option4BP = getBPOption( preOption4BP )
%设置BP的参数
% 输入 BP网络的选项 preOption4BP
% 返回：
% AE网络的选项：option4BP
% decayLambda：	权重衰减系数——正则项罚项权重；
% activation：	激活函数类型；

% isBatchNorm：	是否使用 Batch Normalization 来 speed-up学习速度；

% isDenoising：	是否使用 denoising 规则
% noiseLayer：	AE中添加噪声的层：'firstLayer' or 'allLayers'
% noiseRate：	每一位添加噪声的概率
% noiseMode：	添加噪声的模式：'OnOff' or 'Guass'
% noiseMean：	高斯模式：均值
% noiseSigma：	高斯模式：标准差

    if isfield( preOption4BP, 'decayLambda' )
        option4BP.decayLambda = preOption4BP.decayLambda;
    else
        option4BP.decayLambda = 0.001;
    end
    if isfield( preOption4BP, 'activation' )
        option4BP.activation = preOption4BP.activation;
    else
        option4BP.activation = { 'softmax' };
    end

    % batchNorm
    if isfield( preOption4BP, 'isBatchNorm' )
        option4BP.isBatchNorm = preOption4BP.isBatchNorm;
    else
        option4BP.isBatchNorm = 0;
    end

    % denoising
    if isfield( preOption4BP, 'isDenoising' )
        option4BP.isDenoising = preOption4BP.isDenoising;
		if option4BP.isDenoising
			% denoising每一层 或 只第一个输入层
			if isfield( preOption4BP, 'noiseLayer' )
				option4BP.noiseLayer = preOption4BP.noiseLayer;
			else
				option4BP.noiseLayer = 'firstLayer';
			end
			% 噪声概率
			if isfield( preOption4BP, 'noiseRate' )
				option4BP.noiseRate = preOption4BP.noiseRate;
			else
				option4BP.noiseRate = 0.1;
			end
			% 噪声模式：高斯 或 开关
			if isfield( preOption4BP, 'noiseMode' )
				option4BP.noiseMode = preOption4BP.noiseMode;
			else
				option4BP.noiseMode = 'OnOff';
			end
			switch option4BP.noiseMode
				case 'Guass'
					if isfield( preOption4BP, 'noiseMean' )
						option4BP.noiseMean = preOption4BP.noiseMean;
					else
						option4BP.noiseMean = 0;
					end
					if isfield( preOption4BP, 'noiseSigma' )
						option4BP.noiseSigma = preOption4BP.noiseSigma;
					else
						option4BP.noiseSigma = 0.01;
					end
			end
		end
    else
        option4BP.isDenoising = 0;
    end
end




