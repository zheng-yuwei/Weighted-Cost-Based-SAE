function [ optTheta, bestGlobal, bestGlobalFit ] = optWeightedCostByPSO( runSAEOnce, ...
    architecture, option4PSO )
%利用 粒子群算法PSO 优化fun参数

%% 初始化参数
% 初始化 粒子种群， individualSize * population
individualSize = sum( architecture(1:(end-2)) );
individuals    = rand(  individualSize, option4PSO.population ) * 2; % [0,2]之间
bestSelf       = zeros( individualSize, option4PSO.population ); % 保存每个个体自身的历史最优解
bestGlobal     = zeros( individualSize, 1 ); % 保存种群的历史最优解
% 初始化适应度，越大越好
bestSelfFit    = zeros( 1, option4PSO.population );
bestGlobalFit  = 0;
bestGlobalFit  = 0;
% bestNowGlobalFit = 0;
% 初始化速度
velocity = zeros( individualSize, option4PSO.population );
% 初始化一些参数
wMax = 1; wMin = 0.6; % 惯性因子范围
c1 = 1; c2 = 1; % c1,c2为向 自身历史最优 和 全局历史最优 前进的加速因子，理论上对算法影响不大（解，收敛速度会影响）
% r1 = 0; r2 = 0; % 两个与c1,c2相对于的随机因子，增大搜索随机性

%% PSO迭代搜索
for iter = 1:option4PSO.iteration
    % PSO理论：阻尼e 决定搜索振幅大小（反比），决定了搜索的exploration和exploitation
    % e = ( 1-w ) / [ (c1*r1 + c2*r2) * sqrt(2w + 2 - c1*r1 - c2*r2) ]
    % 所以前期w惯性大，阻尼小，振幅大，搜索范围大（理论，具体还是调参）
    w = wMax - iter * ( wMax - wMin ) / option4PSO.iteration; % 惯性因子
    
    % 求种群的个体适应度，并 更新种群个体
    for pop = 1:option4PSO.population
        [ ~, accuracy ]  = runSAEOnce( individuals( :, pop ) );
        
        % 是否是个体自身的历史最优解
        if accuracy > bestSelfFit( 1, pop )
			bestSelfFit( 1, pop ) = accuracy;
            bestSelf( :, pop )    = individuals( :, pop );
            % 是否是全局最优解
            if accuracy > bestGlobalFit
                bestGlobal    = individuals( :, pop );
                bestGlobalFit = accuracy;
            end
        end
    end
    
    % disp( ['全局最优解为：' num2str(bestGlobalFit * 100) '%'] );
    
    r1 = rand(); r2 = rand();
    velocity = w * velocity + ... % 惯性成分
        c1 * r1 * ( bestSelf - individuals ) + ... % 局部搜索成分
        c2 * r2 * bsxfun( @plus, - individuals, bestGlobal ); % 全局收敛成分
    individuals = individuals + velocity;
end

[ optTheta, bestGlobalFit ]  = runSAEOnce( bestGlobal );

end