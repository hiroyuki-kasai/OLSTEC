function [U,R,err_reg,sub_err] = petrels_tracking(YL,I,J,S,numr,numc,maxrank,maxCycles,lambda,Uinit)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PETRELS by Yuejie Chi
%  modified from GROUSE (Grassman Rank-One Update Subspace Estimation) matrix completion code 
%  by Ben Recht and Laura Balzano, February 2010.
%
%  Given a sampling of entries of a matrix X, try to construct matrices U
%  and R such that U is unitary and UR' approximates X.  This code 
%  implements a stochastic gradient descent on the set of subspaces.
%
%  Inputs:
%       YL: ground truth, to calculate error
%       (I,J,S) index the known entries across the entire data set X. So we
%       know that for all k, the true value of X(I(k),J(k)) = S(k)
%
%       numr = number of rows
%       numc = number of columns
%           NOTE: you should make sure that numr<numc.  Otherwise, use the
%           transpose of X
%       
%       max_rank = your guess for the rank
%
%       step_size = the constant for stochastic gradient descent step size
%
%       maxCycles = number of passes over the data
%
%       Uinit = an initial guess for the column space U (optional)
%
%   Outputs:
%       U and R such that UR' approximates X.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab specific data pre-processing
%

% Form some sparse matrices for easier matlab indexing
values = sparse(I,J,S,numr,numc);
Indicator = sparse(I,J,1,numr,numc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%Main Algorithm
%



if (nargin<10)
    % initialize U to a random r-dimensional subspace 
    U = orth(randn(numr,maxrank)); 
else
    U = Uinit;
end


err_reg = zeros(maxCycles*numc,1);
sub_err = zeros(maxCycles*numc,1);

for outiter = 1:maxCycles,
    
    fprintf('Pass %d...\n',outiter);
    
    % create a random ordering of the columns for the current pass over the
    % data.
    col_order = randperm(numc);
    
    % initialize the covariance matrix and forgetting parameter
    
    Rinv = repmat(100*eye(maxrank),1,numr);

 
for k=1:numc,
  %  k,
    % Pull out the relevant indices and revealed entries for this column
    idx = find(Indicator(:,col_order(k)));
    idxc = find(~Indicator(:,col_order(k)));

    v_Omega = values(idx,col_order(k));
    U_Omega = U(idx,:);    

    
    % Predict the best approximation of v_Omega by u_Omega.  
    % That is, find weights to minimize ||U_Omega*weights-v_Omega||^2
   
    weights = pinv(U_Omega)*v_Omega;
    norm_weights = norm(weights);
    
    % Compute the residual not predicted by the current estmate of U.

    residual = v_Omega - U_Omega*weights;       
    norm_residual = norm(residual);
    
    
    err_reg((outiter-1)*numc + k) = norm_residual;
    
    % This step update Rinv matrix with forgetting parameter lambda
    % for each observed row in U
    %lambda = 1-0.02*exp(-0.001*((outiter-1)*numc+k-1));
    
    % parallel update
    for i=1:length(idx)
        
        
        Tinv = Rinv(:,(idx(i)-1)*maxrank+1:idx(i)*maxrank);
        
        
        ss = Tinv*weights;
    
        Tinv = (Tinv-ss*ss'*1/(lambda+weights'*Tinv*weights));
        

        % update U_omega
    
        U(idx(i),:) = U_Omega(i,:) + lambda^(-1)*residual(i)*weights'*Tinv;
       
    
        Rinv(:,(idx(i)-1)*maxrank+1:idx(i)*maxrank) = Tinv;
    end
    
    Rinv = lambda^(-1)*Rinv;
    

    % calculating the error
    [Uq,Us,Ur] = svd(U,0); 
     
    sub_err((outiter-1)*numc + k) = norm((eye(numr)-Uq*Uq')*YL,'fro')/norm(YL,'fro');
   

     
end

end

% Once we have settled on our column space, a single pass over the data
% suffices to compute the weights associated with each column.  You only
% need to compute these weights if you want to make predictions about these
% columns.
fprintf('Find column weights...');
R = zeros(numc,maxrank);
for k=1:numc,     
    % Pull out the relevant indices and revealed entries for this column
    idx = find(Indicator(:,k));
    v_Omega = values(idx,k);
    U_Omega = U(idx,:);
    % solve a simple least squares problem to populate R
    R(k,:) = (pinv(U_Omega)*v_Omega)';
end

