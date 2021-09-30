function hgw = comp_hgw(parameter_list, mode)
    if mode == 'c'
        func_w_e = @w_c;
    else % mode == 's'
        func_w_e = @w_s;
    end
    n_e = length(parameter_list);
    hgw = zeros(1, n_e);
    for i = 1:n_e
        pi = parameter_list{i};
        %
        if mod(i, 30) == 0
            fprintf('%d %d %d\n', i, n_e, length(pi));
        end
        %
        t = sum(pi);
        s = subset_sum_closest(pi, t/2, length(pi));
        hgw(i) = func_w_e(s, t);
    end
end

function res = w_c(s, t)
    res = s * (t - s);
end

function res = w_s(s, t)
    res = min(s, t - s);
end




















