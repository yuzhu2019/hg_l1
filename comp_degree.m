function degvec = comp_degree(incidence_list, n_v, hgw)
    degvec = zeros(1, n_v);
    for i = 1:length(incidence_list)
        ii = incidence_list{i};
        degvec(ii) = degvec(ii) + hgw(i);
    end
end
