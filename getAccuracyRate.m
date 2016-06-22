function accuracy = getAccuracyRate( predictLabels, labels )
%计算预测准确率
% by 郑煜伟 Aewil 2016-04

% 将预测的概率矩阵中，每列最大概率的值置1，其他置0
predictLabels = bsxfun( @eq, predictLabels, max( predictLabels ) );
% 找出正确label所对应矩阵的位置，并对这些位置的值求均值
indexRow = labels';
indexCol = 1:length(indexRow);
index    = (indexCol - 1) .* size( predictLabels, 1 ) + indexRow;
accuracy = sum( predictLabels(index) )/length(indexRow);

end