% gunzip('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz');
% load covtype.data % covtype 581012 x 55, class = [1,2,3,4,5,6,7]
clear; 
addpath('OneSpectralClustering');
load('covtype.mat');

covertype_cell = cell(1, 7);
cnt = zeros(1, 7);
for i = 1:7
    idx = find(covtype(:, end) == i);
    covertype_cell{i} = covtype(idx, 1:end-1);
    % num of samples in each class: 211840, 283301, 35754, 2747, 9493, 17367, 20510
    cnt(i) = length(idx);
end

%%
careclass = [4, 5];
data = [covertype_cell{careclass(1)}; covertype_cell{careclass(2)}];
N_0 = cnt(careclass(1));
N_1 = cnt(careclass(2));
N = N_0 + N_1;
y = zeros(1, N);
y(N_0+1:end) = 1;

n_v = length(y);

%%
n_bins = 20;
p = linspace(0, 1, n_bins+1);
quat_flag = 10; % the first 10 features are numerical, the rest are categorical
quantile_th = zeros(quat_flag, n_bins+1);

para_p_list = 0:0.1:1.5;
n_para_p = length(para_p_list);
err_rwc_list = zeros(1, n_para_p);
cheeger_rwc_list = zeros(1, n_para_p);
err_sm_list = zeros(1, n_para_p);
cheeger_sm_list = zeros(1, n_para_p);

for ip = 1:n_para_p

    fprintf('------%d------\n', ip);
    para_p = para_p_list(ip);

    n_e = 0;
    R = zeros(n_e, n_v);
    for i = 1:quat_flag
        quantile_th(i, :) = quantile(data(:,i), p);
        quantile_th(i, 1) = quantile_th(i, 1) - 1;
        for j = 1:n_bins
            temp_list = find(data(:,i) <= quantile_th(i,j+1) & data(:,i) > quantile_th(i,j)); 
            if length(temp_list) > 1 && length(temp_list) < n_v % consider |e| >= 2
                n_e = n_e + 1;
                center = median(data(temp_list,i));
                dist = abs(data(temp_list,i)' - center);
                if all(dist == 0)
                    R(n_e, temp_list) = exp(-para_p * dist);
                else
                    R(n_e, temp_list) = exp(-para_p * dist / max(dist)); 
                end
            end
        end
    end

    [n_e, n_v, card, kappa, R, incidence_list, parameter_list, mu] = hg_para(R, para_p, 'c', 'w', 'covtype');

    %% rw-based clique
    f_rwc = hg_rw_laplacian('c');
    tic
    [~, eigvec_rwc, ~] = f_rwc(R, 'std', 2);
    toc
    vmin_rwc = eigvec_rwc(:, 2);

    %% submodular
    f_sm = hg_expansion('c');
    [A_sm, eigvec_sm] = f_sm(incidence_list, parameter_list, n_v);
    A_sm = sparse(A_sm);

    start = vmin_rwc;
    tic
    [vmin_sm, FctValSeq] = computeEigenvectorGeneral(A_sm,start,true,2,false,mu');
    toc

    %% evaluation
    [y_rwc,~,cheeger_rwc,~,~,~] = createClustersGeneral(vmin_rwc,A_sm,true,-1,2,mu);
    err_rwc = comp_err(y', y_rwc);
    fprintf('err_rwc = %.6f, cheeger = %.6f\n', err_rwc, cheeger_rwc);

    [y_sm,~,cheeger_sm,~,~,~] = createClustersGeneral(vmin_sm,A_sm,true,-1,2,mu);
    err_sm = comp_err(y', y_sm);
    fprintf('err_sm = %.6f, cheeger = %.6f\n', err_sm, cheeger_sm);
    
    err_rwc_list(ip) = err_rwc;
    cheeger_rwc_list(ip) = cheeger_rwc;
    err_sm_list(ip) = err_sm;
    cheeger_sm_list(ip) = cheeger_sm;
    
end

% fname = 'results/covtype45_bin20.mat';
% save(fname, 'para_p_list', 'err_rwc_list', 'cheeger_rwc_list', 'err_sm_list', 'cheeger_sm_list');

