function [x,status,history] = l1_ls(A,varargin)
%
% l1-Regularized Least Squares Problem Solver
%
%   l1_ls solves problems of the following form:
%
%       minimize ||A*x-y||^2 + lambda*sum|x_i|,
%
%   where A and y are problem data and x is variable (described below).
%
% CALLING SEQUENCES
%   [x,status,history] = l1_ls(A,y,lambda [,tar_gap[,quiet]])
%   [x,status,history] = l1_ls(A,At,m,n,y,lambda, [,tar_gap,[,quiet]]))
%
%   if A is a matrix, either sequence can be used.
%   if A is an object (with overloaded operators), At, m, n must be
%   provided.
%
% INPUT
%   A       : mxn matrix; input data. columns correspond to features.
%
%   At      : nxm matrix; transpose of A.
%   m       : number of examples (rows) of A
%   n       : number of features (column)s of A
%
%   y       : m vector; outcome.
%   lambda  : positive scalar; regularization parameter
%
%   tar_gap : relative target duality gap (default: 1e-3)
%   quiet   : boolean; suppress printing message when true (default: false)
%
%   (advanced arguments)
%       eta     : scalar; parameter for PCG termination (default: 1e-3)
%       pcgmaxi : scalar; number of maximum PCG iterations (default: 5000)
%
% OUTPUT
%   x       : n vector; classifier
%   status  : string; 'Solved' or 'Failed'
%
%   history : matrix of history data. columns represent (truncated) Newton
%             iterations; rows represent the following:
%            - 1st row) gap
%            - 2nd row) primal objective
%            - 3rd row) dual objective
%            - 4th row) step size
%            - 5th row) pcg iterations
%            - 6th row) pcg status flag
%
% USAGE EXAMPLES
%   [x,status] = l1_ls(A,y,lambda);
%   [x,status] = l1_ls(A,At,m,n,y,lambda,0.001);
%
 
% AUTHOR    Kwangmoo Koh <deneb1@stanford.edu>
% UPDATE    Apr 8 2007
%
% COPYRIGHT 2008 Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd

%------------------------------------------------------------
%       INITIALIZE
%------------------------------------------------------------

% IPM PARAMETERS
MU              = 2;        % updating parameter of t (parameter for updating the barrier t -> at each iteration if the "strenght" of the barrier is ok it is multiplyed by 2 
MAX_NT_ITER     = 400;      % maximum IPM (Newton) iteration (Maximum number of iterations for the newtons method)

% LINE SEARCH PARAMETERS
ALPHA           = 0.01;     % minimum fraction of decrease in the objective (parameter used for the line search) (Richiede che la funzione decresca almeno di questa frazione rispetto a quanto previsto dal gradiente)
BETA            = 0.5;      % stepsize decrease factor (If the step size is too long, we decrease it by multipling it by beta)
MAX_LS_ITER     = 100;      % maximum backtracking line search iteration (max number iter. for line search)



% VARIABLE ARGUMENT HANDLING
% if the second argument is a matrix or an operator, the calling sequence is
%   l1_ls(A,At,y,lambda,m,n [,tar_gap,[,quiet]]))
% if the second argument is a vector, the calling sequence is
%   l1_ls(A,y,lambda [,tar_gap[,quiet]])
if ( (isobject(varargin{1}) || ~isvector(varargin{1})) && nargin >= 6)
    At = varargin{1};
    m  = varargin{2};
    n  = varargin{3};
    y  = varargin{4};
    lambda = varargin{5};
    varargin = varargin(6:end);
    
elseif (nargin >= 3)
    At = A';
    [m,n] = size(A);
    y  = varargin{1};
    lambda = varargin{2};
    varargin = varargin(3:end);
else
    if (~quiet) disp('Insufficient input arguments'); end
    x = []; status = 'Failed'; history = [];
    return;
end

% VARIABLE ARGUMENT HANDLING
t0         = min(max(1,1/lambda),2*n/1e-3);     % Compute a smart initial value for t (if t0 is good it makes the convergence faster)
defaults   = {1e-3,false,1e-3,5000,zeros(n,1),ones(n,1),t0};        % Construct a list of default values (ex. x = zeros, u = ones)
given_args = ~cellfun('isempty',varargin);
defaults(given_args) = varargin(given_args);        % Se l'utente ha passato dei valori di default in input, vengono sovrascritti nella lista inizializzata prima
[reltol,quiet,eta,pcgmaxi,x,u,t] = deal(defaults{:});

f = [x-u;-x-u];     % Calcola il vettore dei vincoli (all'inizio x=0, u=1 => f sarà tutta negativa che significa vincoli rispettati)

% RESULT/HISTORY VARIABLES
% Variabili per salvare l'andamento del grafico errore/tempo
pobjs = [] ; dobjs = [] ; sts = [] ; pitrs = []; pflgs = [];
pobj  = Inf; dobj  =-Inf; s   = Inf; pitr  = 0 ; pflg  = 0 ;

ntiter  = 0; lsiter  = 0; zntiter = 0; zlsiter = 0;
normg   = 0; prelres = 0; dxu =  zeros(2*n,1);

% diagxtx = diag(At*A);
diagxtx = 2*ones(n,1);      % Approximation of the diagonal of the hessian (used for the preconditioned conjiugate gradient) (we use a vector of 2 assuming the columns to be normalized)

% If quei is false, it prints the "intestazione" of the output table
if (~quiet) disp(sprintf('\nSolving a problem of size (m=%d, n=%d), with lambda=%.5e',...
            m,n,lambda)); end
if (~quiet) disp('-----------------------------------------------------------------------------');end
if (~quiet) disp(sprintf('%5s %9s %15s %15s %13s %11s',...
            'iter','gap','primobj','dualobj','step len','pcg iters')); end

%------------------------------------------------------------
%               MAIN LOOP
%------------------------------------------------------------

% Iterative loop for the newton's method
for ntiter = 0:MAX_NT_ITER
    
    z = A*x-y;      % Compute the residual
    
    %------------------------------------------------------------
    %       CALCULATE DUALITY GAP
    %------------------------------------------------------------

    nu = 2*z;       % Candiate for the dual variable

    % The duality theory requires that the duality variable miust satisfy 
    % some contraints => if it is too big, we make it smaller to make it
    % acceptable
    maxAnu = norm(At*nu,inf);
    if (maxAnu > lambda)
        nu = nu*lambda/maxAnu;
    end
    pobj  =  z'*z+lambda*norm(x,1);         % Primal function value
    dobj  =  max(-0.25*nu'*nu-nu'*y,dobj);  % Dual function value
    gap   =  pobj - dobj;                   % Compute the difference between the primal and dual function (if it is 0 we are at the optimum result possible)

    % Save the values
    pobjs = [pobjs pobj]; dobjs = [dobjs dobj]; sts = [sts s];
    pflgs = [pflgs pflg]; pitrs = [pitrs pitr];

    %------------------------------------------------------------
    %   STOPPING CRITERION
    %------------------------------------------------------------
    % If quiet is false it prints the actual state
    if (~quiet) disp(sprintf('%4d %12.2e %15.5e %15.5e %11.1e %8d',...
        ntiter, gap, pobj, dobj, s, pitr)); end

    % If the gap is small enough, under the given tolerance we end
    if (gap/dobj < reltol) 
        status  = 'Solved';
        history = [pobjs-dobjs; pobjs; dobjs; sts; pitrs; pflgs];
        if (~quiet) disp('Absolute tolerance reached.'); end
        %disp(sprintf('total pcg iters = %d\n',sum(pitrs)));
        return;
    end
    %------------------------------------------------------------
    %       UPDATE t
    %------------------------------------------------------------
    
    % If the step size (of the newton method) s is good (>0.5), we increase the value of t
    % We make the algorithm more aggressive, we get closer to the
    % contraints |xi|
    if (s >= 0.5)
        t = max(min(2*n*MU/gap, MU*t), t);
    end

    %------------------------------------------------------------
    %       CALCULATE NEWTON STEP
    %------------------------------------------------------------
    
    % Calcola i termini reciproci derivati dalle derivate della barriera logaritmica −log(u+x) e −log(u−x)
    % Stiamo calcolando le componenti necessarie per costruire il gradiente
    % e l'hessiana della funzione barrirera logaritmica
    % q1 e q2 sono i termini delle derivate prime dei logaritmi
    % d1 e d2 sono it ermini delle derivate seconde (hessiane)
    q1 = 1./(u+x);          q2 = 1./(u-x);
    d1 = (q1.^2+q2.^2)/t;   d2 = (q1.^2-q2.^2)/t;


    % calculate gradient
    % We compute the toal gradient
    % It is composed of 2 parts: The part relative to x (derivative of the
    % fidelity temrm + deriv. of the barrier)
    % part relative to u: deriv. of the regularization term
    gradphi = [At*(z*2)-(q1-q2)/t; lambda*ones(n,1)-(q1+q2)/t];
    
    % calculate vectors to be used in the preconditioner
    % We compute an estimation of the inverse of the hessian
    prb     = diagxtx+d1;
    prs     = prb.*d1-(d2.^2);

    % set pcg tolerance (relative)
    normg   = norm(gradphi);        % Compute the norm of the gradient
    pcgtol  = min(1e-1,eta*gap/min(1,normg));
    % Imposte the tolerance for the solver which is adaptive (when we are
    % far away from the solution) it is not needed to be precise, but as
    % soon we get close to the solution we need to be precise


    if (ntiter ~= 0 && pitr == 0) pcgtol = pcgtol*0.1; end

    % preconditioned conjiugate gradient
    % Solve the sistem: H delta = -Gradient
    [dxu,pflg,prelres,pitr,presvec] = ...
        pcg(@AXfunc_l1_ls,-gradphi,pcgtol,pcgmaxi,@Mfunc_l1_ls,...
            [],dxu,A,At,d1,d2,d1./prs,d2./prs,prb./prs);

    if (pflg == 1) pitr = pcgmaxi; end
    % Split the solution in the components relativ to x and the ones
    % relative to u
    dx  = dxu(1:n);
    du  = dxu(n+1:end);

    %------------------------------------------------------------
    %   BACKTRACKING LINE SEARCH
    %------------------------------------------------------------
    % Once we have the direction, we need to decide how long we have to
    % take the step size s
    phi = z'*z+lambda*sum(u)-sum(log(-f))/t;    % Compute the value of the actual objective function (including the barrier)
    s = 1.0;        % We start initializing the step = 1
    gdx = gradphi'*dxu;     % Directional derivative
    
    for lsiter = 1:MAX_LS_ITER
        % Compute the newx and newu using the step size s
        newx = x+s*dx; newu = u+s*du;

        % Compute the containts (vincoli) newf
        newf = [newx-newu;-newx-newu];
        
        if (max(newf) < 0) % Verify that we are in the domain
            newz   =  A*newx-y;
            % Compute the new value of the function newphi
            newphi =  newz'*newz+lambda*sum(newu)-sum(log(-newf))/t;
            if (newphi-phi <= ALPHA*s*gdx)  % Arjio condition (using parameter alpha)
                break;
            end
        end
        % If the armjio condition is not satisfied we decrease the step
        % size
        s = BETA*s;
    end
    if (lsiter == MAX_LS_ITER) break; end % exit by BLS
        
    x = newx; u = newu; f = newf;
end


%------------------------------------------------------------
%       ABNORMAL TERMINATION (FALL THROUGH)
%------------------------------------------------------------
if (lsiter == MAX_LS_ITER)
    % failed in backtracking linesearch.
    if (~quiet) disp('MAX_LS_ITER exceeded in BLS'); end
    status = 'Failed';
elseif (ntiter == MAX_NT_ITER)
    % fail to find the solution within MAX_NT_ITER
    if (~quiet) disp('MAX_NT_ITER exceeded.'); end
    status = 'Failed';
end
history = [pobjs-dobjs; pobjs; dobjs; sts; pitrs; pflgs];

return;

%------------------------------------------------------------
%       COMPUTE AX (PCG)
%------------------------------------------------------------

% Function to compute the product matrix-vector with the hessian without
% constructing the hessian matrix (it would be very expensive)
function [y] = AXfunc_l1_ls(x,A,At,d1,d2,p1,p2,p3)
%
% y = hessphi*[x1;x2],
%
% where hessphi = [A'*A*2+D1 , D2;
%                  D2        , D1];

n  = length(x)/2;
x1 = x(1:n);
x2 = x(n+1:end);

y = [(At*((A*x1)*2))+d1.*x1+d2.*x2; d2.*x1+d1.*x2];

%------------------------------------------------------------
%       COMPUTE P^{-1}X (PCG)
%------------------------------------------------------------
% Function to apply P^-1 the preconditioned.
% The preconditioned is an approximation of the inverse of the hessian
% matrix
function [y] = Mfunc_l1_ls(x,A,At,d1,d2,p1,p2,p3)
%
% y = P^{-1}*x,
%

n  = length(x)/2;
x1 = x(1:n);
x2 = x(n+1:end);

y = [ p1.*x1-p2.*x2;...
     -p2.*x1+p3.*x2];

