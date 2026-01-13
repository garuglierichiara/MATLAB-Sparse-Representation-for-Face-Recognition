function X = FISTA_solver(Y, D, lambda, maxit, tol)
% FISTA_solver: Function to implement the Fast Iterative thresholding algorithm (FISTA)
%
% Solve the optimization problem: min 1/2*||D*x-y||_2^2 + lambda*||x||_1
%
% OBS: If Y is a matrix we solve the optimization problem column by columns
%
% INPUT:
%   Y: Matrix of test dataset (Each column is an image) [pixels x n_test_samples]=[m x N]
%   D: Dictionary/Training dataset [pixels x n_train]=[m x n_atoms]
%   lambda: regularization parameter
%   maxit: maximum number of iterations
%   tol: tolerance for stopping criterion
%
% OUTPUT:
%   X : sparse matrix of coefficients [n_atoms x n_samples]= [n x N]


n_atoms = size(D,2);      % n_atoms = number of atoms of the dictionary
n_samples = size(Y, 2);

% Compute the Step Size:
% The step size must be <= 1/L with L lipsitz constant of Gradient of f(x)
L = 2 * eigs(D' * D, 1); % use eigs to find the biggest eigenvalue
invL = 1 / L;

% Pre-compute the matrices we will need for computing the gradient of f(x):
DtD = D' * D;
DtY = D' * Y;

% Initialization:
X_old = zeros(n_atoms, n_samples);
Y_old = X_old;      % Alpha ???
mu_old = 1;

% OBS: It is not necessary to do a loop over the column, because the
% funzion we are optimizing is separable => so we are actually solving the
% problem for the columns of Y separately
for k = 1:maxit
    
    % STEP 1
    % Gradient of f(x)= 1/2*||DX - Y||^2 is: 
    % D'*(D*X - Y) = D'D*X - D'Y
    % The gradient is evaluated at Y_old
    grad = DtD * Y_old - DtY;
    
    % Gradient Descent step (For the smooth part)
    Z = Y_old - invL * grad;
    
    % Proximal Operator (Soft Thresholding)
    threshold = lambda * invL;
    X_new = sign(Z) .* max(abs(Z) - threshold, 0);
    
    % Stopping criterion
    % If X changes only a little bit, we stop
    norm_Xold = max(norm(X_old, 'fro'), 1e-10);
    
    if norm(X_new - X_old, 'fro') / norm_Xold < tol
        break;
    end

    % STEP 2
    % Update the momentum (Nesterov Acceleration)
    mu_new = (1 + sqrt(1 + 4 * mu_old^2)) / 2;
    Y_old = X_new + ((mu_old - 1) / mu_new) * (X_new - X_old);
    
    % STEP 3
    % Update the variables
    X_old = X_new;
    mu_old = mu_new;
end

X = X_new;

end