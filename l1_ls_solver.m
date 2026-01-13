function X = l1_ls_solver(Y, D, lambda, rel_tol)
% L1_LS_SOLVER: Function to implement the solver l1_ls (Boyd et al.)
%               for each column y 
%
% Solve the optimization Pb: min ||y - Dx||_2^2 + lambda*||x||_1
%
% INPUT:
%   Y: Matrix of test dataset (Each column is an image) [pixels x n_test_samples
%   D: Dictionary/Training dataset [pixels x n_train]
%   lambda: regularization parameter
%   rel_tol: relative tolerance
%
% OUTPUT:
%   X: Sparse matrix of coefficients [n_atoms x n_samples]

    % quiet: true to not print messages during the execution
    quiet = true;   

    [m, n] = size(D);
    n_test = size(Y, 2);
    
    % Pre-allocate the matrix where to save the coefficients
    X = zeros(n, n_test);

    % l1_ls takes in input a single column vector y  
    % => Add a loop over the column of Y
    for i = 1:n_test
        y = Y(:, i);
        
        % Call the original solver
        [x,status] = l1_ls(D, y, lambda, rel_tol, quiet);        
        X(:, i) = x;
    end
end