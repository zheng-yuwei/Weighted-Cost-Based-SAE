function option4BPNN = getBPNNOption( preOption4BPNN )
%设置BP的参数
% by 郑煜伟 Aewil 2016-04
% 输入 BP网络的选项 preOption4BPNN
% 返回：
% AE网络的选项： option4BPNN
% decayLambda：  权重衰减系数——正则项罚项权重；
% activation：   激活函数类型；

% isBatchNorm：  是否使用 Batch Normalization 来 speed-up学习速度；

% isDenoising：  是否使用 denoising 规则
% noiseLayer：  	AE中添加噪声的层：'firstLayer' or 'allLayers'
% noiseRate：    每一位添加噪声的概率
% noiseMode：   	添加噪声的模式：'OnOff' or 'Guass'
% noiseMean：   	高斯模式：均值
% noiseSigma：  	高斯模式：标准差

if isfield( preOption4BPNN, 'decayLambda' )
	option4BPNN.decayLambda = preOption4BPNN.decayLambda;
else
	option4BPNN.decayLambda = 0.001;
end

if isfield( preOption4BPNN, 'activation' )
	option4BPNN.activation = preOption4BPNN.activation;
else
	error( '激活函数列表必须由你自己来定！' );
end

% batchNorm
if isfield( preOption4BPNN, 'isBatchNorm' )
	option4BPNN.isBatchNorm = preOption4BPNN.isBatchNorm;
else
	option4BPNN.isBatchNorm = 0;
end

% denoising
if isfield( option4BPNN, 'isDenoising' )
    option4BPNN.isDenoising = option4BPNN.isDenoising;
    if option4BPNN.isDenoising
        % denoising每一层 或 只第一个输入层
        if isfield( option4BPNN, 'noiseLayer' )
            option4BPNN.noiseLayer = option4BPNN.noiseLayer;
        else
            option4BPNN.noiseLayer = 'firstLayer';
        end
        % 噪声概率
        if isfield( option4BPNN, 'noiseRate' )
            option4BPNN.noiseRate = option4BPNN.noiseRate;
        else
            option4BPNN.noiseRate = 0.1;
        end
        % 噪声模式：高斯 或 开关
        if isfield( option4BPNN, 'noiseMode' )
            option4BPNN.noiseMode = option4BPNN.noiseMode;
        else
            option4BPNN.noiseMode = 'OnOff';
        end
        switch option4BPNN.noiseMode
            case 'Guass'
                if isfield( option4BPNN, 'noiseMean' )
                    option4BPNN.noiseMean = option4BPNN.noiseMean;
                else
                    option4BPNN.noiseMean = 0;
                end
                if isfield( option4BPNN, 'noiseSigma' )
                    option4BPNN.noiseSigma = option4BPNN.noiseSigma;
                else
                    option4BPNN.noiseSigma = 0.01;
                end
        end
    end
else
    option4BPNN.isDenoising = 0;
end

end