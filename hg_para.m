function [n_e, n_v, card, kappa, R, incidence_list, parameter_list, mu] = hg_para(R, para_p, mode, mu_w, dataset)

    [n_e, n_v] = size(R);
    fprintf('number of nodes/hyperedges: %d, %d\n', n_v, n_e);
    
    card = sum(R > 0, 2); % edge cardinality
    fprintf('max/min edge cardinality %d, %d\n', max(card), min(card));
        
    if strcmp(dataset, '20news')
        if para_p == 0 
            R = double(R > 0); % cardinality-based
        else
            R = R.^para_p;
        end
    end
    
    kappa = std(R, 1, 2);
%     kappa = ones(1, n_e);

    incidence_list = cell(1, n_e);
    parameter_list = cell(1, n_e);
    w_max = 0;
    w_min = inf;
    for e_i = 1:n_e
        incidence_list{e_i} = find(R(e_i, :) > 0);
        if mode == 'c'
            parameter_list{e_i} = R(e_i, incidence_list{e_i}) * (kappa(e_i)^0.5);
        else % mode == 's'
            parameter_list{e_i} = R(e_i, incidence_list{e_i}) * kappa(e_i);
        end
        w_max = max(w_max, max(parameter_list{e_i}));
        w_min = min(w_min, min(parameter_list{e_i}));
    end
    
    fprintf('max/min EDVWs %.5f, %.5f\n', w_max, w_min);
    
    if strcmp(mu_w, 'u')
        % vertex weight = number of incident edges
        mu = comp_degree(incidence_list, n_v, ones(1, n_e));
    else
        hgw = comp_hgw(parameter_list, mode);
        mu = comp_degree(incidence_list, n_v, hgw);
    end
    
end

