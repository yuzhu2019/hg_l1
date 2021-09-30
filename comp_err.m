function [err] = comp_err(y, y_est)
    err = sum(y == y_est)/length(y);
    err = min(err, 1-err);
end

