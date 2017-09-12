function [Xsol, infos, sub_infos] = TeCPSGD(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options)
% TeSPSGD algorithm.
%
% Inputs:
%       A_in            full tensor data to be tracked.
%       Omega_in        logical data of traing tensor set to speficy observable/missing elements.
%       Gamma_in        logical data of test tensor set to speficy observable/missing elements.
%       tensor_dims     dimension of tensor.
%       rank            max rank.
%       xinit           initial tensor data.
%       options         structure data of options.
% Output:
%       XSol            solution.
%       infos           information.
%       sub_infos       sub information.
%
% Reference:
%       M. Mardani, G. Mateos, and G.B. Giannakis, 
%       "Subspace learning and imputation for streaming big data matrices and tensors," 
%       IEEE Transactions on Signal Processing, vol. 63, no. 10, pp. 266-2677, 2015.
%
%                   
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017


    A               = A_in;             % Full entries
    Omega           = Omega_in;         % Training set 'Omega'
    Gamma           = Gamma_in;         % Test set 'Gamma'
    
    %
    A_Omega         = Omega_in.*A_in;   % Training entries i.e., Omega_in.*A_in   
    if ~isempty(Gamma_in)
        A_Gamma         = Gamma_in.*A_in;   % Test entries i.e., Gamma_in.*A_in
    else 
        A_Gamma     = [];
    end
 
    if isempty(xinit)
        A_t0 = randn(tensor_dims(1), rank);
        B_t0 = randn(tensor_dims(2), rank);        
        C_t0 = randn(tensor_dims(3), rank);        
    else
        A_t0 = xinit.A;
        B_t0 = xinit.B;        
        C_t0 = xinit.C;
    end
    

    % set tensor size
    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    slice_length    = tensor_dims(3);

    
    % set options
    lambda          = options.lambda;
    mu              = options.mu;
    stepsize_init   = options.stepsize;
    maxepochs       = options.maxepochs;
    tolcost         = options.tolcost;    
    store_subinfo   = options.store_subinfo;
    store_matrix    = options.store_matrix; 
    verbose         = options.verbose;
    
    if ~isfield(options, 'permute_on')
        permute_on = 1;
    else
        permute_on = options.permute_on;
    end 

    
    % calculate initial cost
    Rec = zeros(rows, cols, slice_length);
    for k=1:slice_length
        gamma = C_t0(k,:)';
        Rec(:,:,k) = A_t0 * diag(gamma) * B_t0';
    end    
    train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
    if ~isempty(Gamma) && ~isempty(A_Gamma)
        test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
    else
        test_cost = 0;
    end
    

    % initialize infos
    infos.iter = 0;
    infos.train_cost = train_cost;
    infos.test_cost = test_cost;
    infos.time = 0;
    
    % initialize sub_infos
    sub_infos.inner_iter = 0;
    sub_infos.err_residual = 0;
    sub_infos.err_run_ave = 0;
    sub_infos.global_train_cost = 0; 
    sub_infos.global_test_cost = 0;  
    if store_matrix
        sub_infos.I = zeros(rows * cols, slice_length);
        sub_infos.L = zeros(rows * cols, slice_length);
        sub_infos.E = zeros(rows * cols, slice_length);
    end     
    
    % set parameters
    eta = 0;
    
    if verbose > 0
        fprintf('TeCPSGD [%d] Epoch 000, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, train_cost, test_cost, eta);
    end    
    

    % main loop
    for outiter = 1 : maxepochs

        % permute samples
        if permute_on
            col_order = randperm(slice_length);
        else
            col_order = 1:slice_length;
        end

        % Begin the time counter for the epoch
        t_begin = tic();    

        for k=1:slice_length
            
            fnum = (outiter - 1) * slice_length + k;
            
            % sampled original image
            I_mat = A(:,:,col_order(k));
            Omega_mat = Omega(:,:,col_order(k));
            I_mat_Omega = Omega_mat .* I_mat;               

            % Reculculate gamma (C)
            temp3 = 0;
            temp4 = 0;
            for m=1:rows
                alpha_remat = repmat(A_t0(m,:)', 1, cols);
                alpha_beta = alpha_remat .* B_t0';
                I_row = I_mat_Omega(m,:);
                temp3 = temp3 + alpha_beta * I_row';

                Omega_mat_ind = find(Omega_mat(m,:));
                alpha_beta_Omega = alpha_beta(:,Omega_mat_ind);
                temp4 = temp4 + alpha_beta_Omega * alpha_beta_Omega';
            end
            temp4 = lambda * eye(rank) + temp4;
            gamma = temp4 \ temp3;                                             % equation (18)               

            L_rec = A_t0 * diag(gamma) * B_t0';
            diff = Omega_mat.*(I_mat - L_rec);

            if 0
                eta = 1/mu;
                %A_t1 = (1 - lambda/(fnum*mu)) * A_t0 + 1/mu * diff	* B_t0 * diag(gamma);   % equation (20)&(21)
                %B_t1 = (1 - lambda/(fnum*mu)) * B_t0 + 1/mu * diff' * A_t0 * diag(gamma);  % equation (20)&(22)
                A_t1 = (1 - lambda*eta/fnum) * A_t0 + eta * dif   * B_t0 * diag(gamma);   % equation (20)&(21)
                B_t1 = (1 - lambda*eta/fnum) * B_t0 + eta * diff' * A_t0 * diag(gamma);  % equation (20)&(22)                
            else
                eta = stepsize_init/(1+lambda*stepsize_init*fnum);
                A_t1 = (1 - lambda*eta) * A_t0 + eta * diff *  B_t0 * diag(gamma);   % equation (20)&(21)
                B_t1 = (1 - lambda*eta) * B_t0 + eta * diff' * A_t0 * diag(gamma);  % equation (20)&(22)                 
            end

            % Reculculate weights
            %weights = pinv(A_t1) * I_mat_Omega * pinv(B_t1');
            %t = diag(weights);

            % Update of A and B
            A_t0 = A_t1;
            B_t0 = B_t1; 

            % Reculculate gamma (C)
            temp3 = 0;
            temp4 = 0;
            for m=1:rows
                alpha_remat = repmat(A_t0(m,:)', 1, cols);
                alpha_beta = alpha_remat .* B_t0';
                I_row = I_mat_Omega(m,:);
                temp3 = temp3 + alpha_beta * I_row';

                Omega_mat_ind = find(Omega_mat(m,:));
                alpha_beta_Omega = alpha_beta(:,Omega_mat_ind);
                temp4 = temp4 + alpha_beta_Omega * alpha_beta_Omega';
            end
            temp4 = lambda * eye(rank) + temp4;
            gamma = temp4 \ temp3;                                             % equation (18)                
            
            % Store gamma into C_t0       
            C_t0(col_order(k),:) = gamma';            

            % Reconstruct Low-rank Matrix
            L_rec = A_t0 * diag(gamma) * B_t0';
%             if disp_flag            
%                 L{alg_idx} = [L{alg_idx} L_rec(:)];
%             end   

            if store_matrix
                E_rec = I_mat - L_rec;
                %sub_infos.E = [sub_infos.E E_rec(:)]; 
                sub_infos.I(:,k) = I_mat_Omega(:);
                sub_infos.L(:,k) = L_rec(:);
                sub_infos.E(:,k) = E_rec(:);
            end

            if store_subinfo
                % Residual Error
                norm_residual   = norm(I_mat(:) - L_rec(:));
                norm_I          = norm(I_mat(:));
                error           = norm_residual/norm_I; 
                sub_infos.inner_iter    = [sub_infos.inner_iter (outiter-1)*slice_length+k];            
                sub_infos.err_residual    = [sub_infos.err_residual error];  

                % Running-average Estimation Error
                if k == 1
                    run_error   = error;
                else
                    run_error   = (sub_infos.err_run_ave(end) * (k-1) + error)/k;
                end
                sub_infos.err_run_ave     = [sub_infos.err_run_ave run_error];            

                % Store reconstruction error
                if store_matrix
                    E_rec = I_mat - L_rec;
                    sub_infos.E = [sub_infos.E E_rec(:)]; 
                end

                for f=1:slice_length
                    gamma = C_t0(f,:)';
                    Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
                end 

                % Global train_cost computation
                train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
                if ~isempty(Gamma) && ~isempty(A_Gamma)
                    test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
                else
                    test_cost = 0;
                end    
                sub_infos.global_train_cost  = [sub_infos.global_train_cost train_cost]; 
                sub_infos.global_test_cost  = [sub_infos.global_test_cost test_cost];             

                if verbose > 1
                    fnum = (outiter-1)*slice_length + k;
                    fprintf('TeCPSGD: fnum = %03d, cost = %e, error = %e\n', fnum, train_cost, error);
                end
            end
        end

        
        % store infos
        infos.iter = [infos.iter; outiter];
        infos.time = [infos.time; infos.time(end) + toc(t_begin)];        
        
        if ~store_subinfo
            for f=1:slice_length
                gamma = C_t0(f,:)';
                Rec(:,:,f) = A_t0 * diag(gamma) * B_t0';
            end 
                
            train_cost = compute_cost_tensor(Rec, Omega, A_Omega, tensor_dims);
            if ~isempty(Gamma) && ~isempty(A_Gamma)
                test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
            else
                test_cost = 0;
            end            
        end
        infos.train_cost = [infos.train_cost; train_cost];
        infos.test_cost = [infos.test_cost; test_cost];        

        if verbose > 0
            fprintf('TeCPSGD [%d] Epoch %0.3d, Cost %7.3e, Cost(test) %7.3e, Stepsize %7.3e\n', stepsize_init, outiter, train_cost, test_cost, eta);
        end
        
        % stopping criteria: cost tolerance reached
        if train_cost < tolcost
            fprintf('train_cost sufficiently decreased.\n');
            break;
        end   
    end




% Once we have settled on our column space, a single pass over the data
% suffices to compute the weights associated with each column.  You only
% need to compute these weights if you want to make predictions about these
% columns.
% fprintf('Find column weights...');
% R = zeros(numc,maxrank);
% for k=1:numc,     
%     % Pull out the relevant indices and revealed entries for this column
%     idx = find(Indicator(:,k));
%     v_Omega = values(idx,k);
%     U_Omega = U(idx,:);
%     % solve a simple least squares problem to populate R
%     R(k,:) = (U_Omega\v_Omega)';
% end

    Xsol.A = A_t0;
    Xsol.B = B_t0;    
    Xsol.C = C_t0;
end




