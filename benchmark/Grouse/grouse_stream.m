function [ U, weights, err_redidual, err_subspace ] = grouse_stream( U_gt, y_Omega, idx, U0, numr, numc, step_size, initer, outiter )

    has_u_gt = 0;
    norm_U_gt = norm(U_gt);
    if norm_U_gt
        has_u_gt = 1;
    end
    
    err_redidual = 0;
    err_subspace = 0;
    U = U0;
    v_Omega = y_Omega;
    U_Omega = U(idx,:);   
    
    % Predict the best approximation of v_Omega by u_Omega.  
    % That is, find weights to minimize ||U_Omega*weights-v_Omega||^2
    weights = U_Omega\v_Omega;
    norm_weights = norm(weights);

    % Compute the residual not predicted by the current estmate of U.

    residual = v_Omega - U_Omega*weights;   
    norm_residual = norm(residual);

    % This step-size rule is given by combining Edelman's geodesic
    % projection algorithm with a diminishing step-size rule from SGD.  A
    % different step size rule could suffice here...        

    sG = norm_residual*norm_weights;
    err_redidual = norm_residual/norm(v_Omega);
    t = step_size*sG/( (outiter-1)*numc + initer);

    % Take the gradient step.    
    if t<pi/2, % drop big steps        
        alpha = (cos(t)-1)/norm_weights^2;
        beta = sin(t)/sG;

        step = U*(alpha*weights);
        step(idx) = step(idx) + beta*residual;

        U = U + step*weights';
    end 


    % Calculate a final weights
    % solve a simple least squares problem
    U_Omega = U(idx,:);
    weights = U_Omega\v_Omega;
    
    % calculating the error
    if has_u_gt
        [Uq,Us,Ur] = svd(U,0);        
        err_subspace = norm((eye(numr)-Uq*Uq')*U_gt)/norm_U_gt;
    end
    
end

