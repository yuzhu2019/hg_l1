function func = hg_expansion(mode)
    if strcmp(mode, 'c')
        func = @clique_expansion;
    else % strcmp(mode, 's')
        func = @star_expansion;
    end
end

function [A, eigvec] = clique_expansion(incidence_list, parameter_list, n_v)
    n_e = length(incidence_list);
    A = zeros(n_v);
    for e_i = 1:n_e
        nodes = incidence_list{e_i};
        EDVWs = parameter_list{e_i};
        esize = length(nodes);
        for i = 1:esize
            vi = nodes(i);
            for j = i+1:esize
                vj = nodes(j);
                A(vi, vj) = A(vi, vj) + EDVWs(i) * EDVWs(j);
                A(vj, vi) = A(vi, vj);
            end
        end
    end
    D = sum(A).^(-0.5);
    [eigvec, ~] = eigs(D' .* A .* D, 2, 'largestreal');
    eigvec = eigvec(:, 2); % second smallest eigval of normalized L
end
