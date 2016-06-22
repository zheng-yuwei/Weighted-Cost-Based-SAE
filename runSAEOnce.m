function [ optTheta, accuracy ] = runSAEOnce( images4Train, labels4Train, ...
    images4Test, labels4Test, ... % 数据
    architecture, ...
    option4SAE, option4BPNN, ...
    isDispNetwork, isDispInfo, weightedCost, ~ )
%设置SAE参数 并 运行一次 SAE
% by 郑煜伟 Aewil 2016-04

if exist( 'weightedCost', 'var' )
    option4SAE.option4AE.weightedCost = weightedCost;
end
%% 训练SAE
theta4SAE = trainSAE( images4Train, labels4Train, architecture, option4SAE ); % 训练SAE
if isDispNetwork
    % 展示网络中间层所抽取的feature
    displayNetwork( reshape(theta4SAE(1 : 784 * 400), 400, 784) );
    displayNetwork( (reshape(theta4SAE(1 : 784 * 400), 400, 784)' * ...
        reshape(theta4SAE(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')' );
end
if isDispInfo
    % 用 未微调的SAE参数 进行预测
    predictLabels = predictNN( images4Train, architecture, theta4SAE, option4BPNN );
    accuracy = getAccuracyRate( predictLabels, labels4Train );
    disp( ['MNIST训练集 SAE(未微调）准确率为： ', num2str(accuracy * 100), '%'] );
    
    predictLabels = predictNN( images4Test, architecture, theta4SAE, option4BPNN );
    accuracy = getAccuracyRate( predictLabels, labels4Test );
    disp( ['MNIST测试集 SAE(未微调）准确率为： ', num2str(accuracy * 100), '%'] );
end

%% BP fine-tune
[ optTheta, ~ ] = trainBPNN( images4Train, labels4Train, theta4SAE, architecture, option4BPNN );
if isDispNetwork
    % 展示网络中间层所抽取的feature
    displayNetwork( reshape(optTheta(1 : 784 * 400), 400, 784) );
    displayNetwork( (reshape(optTheta(1 : 400 * 784), 400, 784)' * ...
        reshape(optTheta(784 * 400 + 1 : 784 * 400 + 400*200 ), 200, 400)')' );
end
%% 用 fine-tune后SAE 进行预测
if isDispInfo
    predictLabels = predictNN( images4Train, architecture, optTheta, option4BPNN );
    accuracy = getAccuracyRate( predictLabels, labels4Train );
    disp( ['MNIST训练集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%'] );
end
predictLabels = predictNN( images4Test, architecture, optTheta, option4BPNN );
accuracy = getAccuracyRate( predictLabels, labels4Test );
disp( ['MNIST测试集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%'] ); % pppppppppppppppppppppppppppppppppppppppp
if isDispInfo
    disp( ['MNIST测试集 SAE(微调）准确率为： ', num2str(accuracy * 100), '%'] );
end

end


