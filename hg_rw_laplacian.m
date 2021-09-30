function func = hg_rw_laplacian(mode)
    if strcmp(mode, 'c')
        func = @clique_laplacian;
    else % strcmp(mode, 's')
        func = @star_lapalcian;
    end
end

function [T, eigvec, eigval] = clique_laplacian(R, hgw, k)
% paper: Hypergraph Random Walks, Laplacians, and Clustering
% T: I - normalized L, equation (15)
% eigvec, eigval: smallest k magnitude eigvalues of normalized L
    n_e = size(R, 1);
    if strcmp(hgw, 'std')
        hyperedge_weights = std(R, 1, 2)'; 
    else % strcmp(hgw, 'u')
        hyperedge_weights = ones(1, n_e);
    end
    W = (R' > 0) .* hyperedge_weights; 
    P = (1./sum(W, 2)) .* W .* (1./sum(R, 2))' * R; 
    [pi, ~] = eigs(P', 1);
    assert(max(abs(pi' * P - pi')) < 1e-8);
    pi = pi./sum(pi); 
    pi_sqrt = sqrt(pi);
    T = pi_sqrt .* P .* (1./pi_sqrt)';
    T = (T + T')/2;
    [eigvec, eigval] = eigs(T, k, 'largestreal');
    eigval = 1 - diag(eigval);
    assert(abs(eigval(1)) < 1e-8);
end
