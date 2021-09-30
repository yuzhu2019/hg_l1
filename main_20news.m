clear; 
addpath('OneSpectralClustering');
   
para_p_list = 0:0.05:0.4;

n_para_p = length(para_p_list);
err_rwc_list = zeros(1, n_para_p);
cheeger_rwc_list = zeros(1, n_para_p);
err_sm_list = zeros(1, n_para_p);
cheeger_sm_list = zeros(1, n_para_p);

for ip = 1:n_para_p

    fprintf('------%d------\n', ip);
    load('python/20news_10.mat'); % R: |E|x|V|, y: |V|x1
    para_p = para_p_list(ip);
    [n_e, n_v, card, kappa, R, incidence_list, parameter_list, mu] = hg_para(R, para_p, 'c', 'w', '20news');

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

% fname = 'results/20news_10.mat';
% save(fname, 'para_p_list', 'err_rwc_list', 'cheeger_rwc_list', 'err_sm_list', 'cheeger_sm_list');





















