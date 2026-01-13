function  [X,Lset] = omp_solver(Y,D,maxit,tol)
% function  [X,Lset] = omp_solverok(Y,D,maxit,tol)
% function that implements the Matching Pursuit algorithm to compute a
% sparse x such that
%
% Y \approx D*x with ||x||_0 <= maxit
%
% INPUT
% Y: target vector
% D: dictionary
% maxit: upper bound on the number of nonzero entries of x
% tol: threshold on the residual norm

% initialization
normy = norm(Y);
X = sparse(size(D,2),size(Y,2));

% work column-wise on Y
for i=1:size(Y,2)

    Lset{i}=[];
    normres2=norm(Y(:,i))^2; %normres2=norm(Res(:,i))^2;
    p = D'*Y(:,i);  %Res(:,i) = Y(:,i);

    % start the loop to fill the components of x one at the time
    for k=1:maxit

        % p = D'*Res(:,i);    % this is updated at the bottom

        % don't look at the indeces we've already selected
        p(Lset{i})=0;
        % get the index of the max
        [~,ii]=max(abs(p(:)));

        % update the index set
        Lset{i}=[Lset{i},ii];

        % Gram Schmidt step (to avoid computing from scratch a QR at each
        % iteration)
        if k==1
            Q(:,1)=D(:,ii)/norm(D(:,ii));    
        else
            % (Classical GS step performed twice for stability)
            Q(:,k)=D(:,ii)-Q(:,1:k-1)*(Q(:,1:k-1)'*D(:,ii));
            Q(:,k)=Q(:,k)-Q(:,1:k-1)*(Q(:,1:k-1)'*Q(:,k));
            Q(:,k)=Q(:,k)/norm(Q(:,k));

        end
  
        proj(k,1)=Q(:,k)'*Y(:,i);

        % Res(:,i)=Res(:,i)-Q(:,k)*proj(k);   % This would be the residual update
        normres2=normres2-proj(k)^2;
        p = p - (D'*Q(:,k))*proj(k);
  
        % normres(i,k)=norm(Res);
        normres(i,k)=sqrt(normres2);

        if normres(i,k)/normy<tol 
            break
        end
    end

    % at the end we need to compute x To do so we first retrieve the R
    % factor of the qr
    R=triu(Q'*D(:,Lset{i}));
    % and then solve the linear system
    X(Lset{i},i)=R\proj;

    % checking the final residual, only for debugging
    %Res(:,i)=Y(:,i)-D(:,Lset{i})*X(Lset{i},i);

end




