function [optTheta, cost] = trainAE( input, theta, architecture, countAE, option4AE )
%训练AE网络
% by 郑煜伟 Aewil 2016-04

% 函数 calcAEBatch 可以根据当前点计算 cost 和 gradient，但是步长不确定
% 这里，调用Mark Schmidt的包来优化迭代 步长：用了l-BFGS
% Mark Schmidt (http://www.di.ens.fr/~mschmidt/Software/minFunc.html) [仅供学术]
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 100;	  % L-BFGS 的最大迭代代数
options.display = 'off';
% options.TolX = 1e-3;

% 判断该 countAE层 AE是否需要添加noise 以 使用denoising规则
[ isDenoising, inputCorrupted ] = denoisingSwitch( input, countAE, option4AE );
if isDenoising
	[optTheta, cost] = minFunc( @(x) calcAEBatch( input, x, architecture, option4AE, inputCorrupted ), ...
            theta, options);
else
	[optTheta, cost] = minFunc( @(x) calcAEBatch( input, x, architecture, option4AE ), ...
            theta, options);
end

end

function [ isDenoising, inputCorrupted ] = denoisingSwitch( input, countAE, option4AE )
%判断该层AE是否需要添加noise以使用denoising规则
% 返回 是否isDenoising的标志 及 噪声

% isDenoising：	是否使用 denoising 规则
% noiseLayer：	AE中添加噪声的层：'firstLayer' or 'allLayers'
% noiseRate：	每一位添加噪声的概率
% noiseMode：	添加噪声的模式：'OnOff' or 'Guass'
% noiseMean：	高斯模式：均值
% noiseSigma：	高斯模式：标准差

    isDenoising    = 0;
    inputCorrupted = [];
    if option4AE.isDenoising
        switch option4AE.noiseLayer
            case 'firstLayer'
                if countAE == 1
                    isDenoising = 1;
                end
            case 'allLayers'
                isDenoising = 1;
            otherwise
                error( '错误的AE噪声层数！' );
        end
        if isDenoising
            inputCorrupted = input;
            indexCorrupted = rand( size(input) ) < option4AE.noiseRate;
            switch option4AE.noiseMode
                case 'Guass'
                    % 均值为 noiseMean，标准差为 noiseSigma 的高斯噪声
                    noise = option4AE.noiseMean + ...
                        randn( size(input) ) * option4AE.noiseSigma;
                    noise( ~indexCorrupted ) = 0;
                    inputCorrupted = inputCorrupted + noise;
                case 'OnOff'
                    inputCorrupted( indexCorrupted ) = 0;
            end
        end
    end
end

