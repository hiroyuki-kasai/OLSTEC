function [Xsol, infos, sub_infos] = cp_wopt_mod(A_in, Omega_in, Gamma_in, tensor_dims, rank, xinit, options)
% Interface file for CP-WOPT algorithm
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
%                   
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017


    A               = A_in;
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
    store_subinfo   = options.store_subinfo;
    store_matrix    = options.store_matrix; 
    verbose         = options.verbose;
    
    
    % set an example problem with missing data
    X = tensor(Omega .* A(:,:,1:slice_length));
    P = tensor(Omega);

    if isempty(xinit)
        % Create initial guess using 'nvecs'
        M_init = create_guess('Data', X, 'Num_Factors', rank, 'Factor_Generator', 'nvecs');
    else
        M_init = cell(3,1);
        M_init{1} = xinit.A;
        M_init{2} = xinit.B;       
        M_init{3} = xinit.C;          
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
    infos.iter = 1;
    infos.train_cost = train_cost;
    infos.test_cost = test_cost;
    infos.time = 0;
    
    % initialize sub_infos
    sub_infos.inner_iter = 0;
    sub_infos.err_residual = 0;
    sub_infos.err_run_ave = 0;
    sub_infos.E = zeros(rows, cols);
    sub_infos.global_train_cost = 0; 
    sub_infos.global_test_cost = 0;  
    


    % set up the optimization parameters
    % Get the defaults
    ncg_opts = ncg('defaults');
    % Tighten the stop tolerance (norm of gradient). This is often too large.
    ncg_opts.StopTol = 1.0e-6;
    % Tighten relative change in function value tolearnce. This is often too large.
    ncg_opts.RelFuncTol = 1.0e-9;
    % Increase the number of iterations.
    %ncg_opts.MaxIters = 3*10^2;
    ncg_opts.MaxIters = options.maxepochs;
    % Only display every 10th iteration
    %ncg_opts.DisplayIters = 10;
    ncg_opts.DisplayIters = options.display_iters;
    % Display the final set of options
    %ncg_opts

    
    % Begin the time counter for the epoch
    t_begin = tic();
       
    % Main routine
    [M, xsol, output] = cp_wopt(X, P, rank, 'init', M_init, 'alg', 'ncg', 'alg_options', ncg_opts);  
    
    if store_subinfo
        L_rec_all = double(full(M));
        for fnum= 1 : slice_length
            % Extract a noiseless original slice
            I_mat_Noiseless = A(:,:,fnum);

            % Extract a reconstructed slice
            L_rec = L_rec_all(:,:,fnum);

    %         if disp_flag
    %             L{alg_idx} = [L{alg_idx} L_rec(:)];
    %         end

            norm_residual   = norm(I_mat_Noiseless(:) - L_rec(:));
            norm_I          = norm(I_mat_Noiseless(:));
            error           = norm_residual/norm_I;  
            sub_infos.inner_iter    = [sub_infos.inner_iter fnum];            
            sub_infos.err_residual  = [sub_infos.err_residual error];          

            % Running-average Estimation Error
            if fnum == 1
                run_error   = error;
            else
                run_error   = (sub_infos.err_run_ave(end) * (fnum-1) + error)/k;
            end
            sub_infos.err_run_ave     = [sub_infos.err_run_ave run_error];             

    %         if disp_flag
    %             % Reconstruct Error Matrix
    %             E_rec = I_mat_Noiseless - L_rec;
    %             E{alg_idx} = [E{alg_idx} E_rec(:)]; 
    %         end  

            if verbose > 1
                fprintf('CP-WOPU: fnum = %03d, error = %e\n', fnum, error);
            end
        end
    end
    
    
    % store infos
    infos.iter = [infos.iter; 2];
    infos.time = [infos.time; infos.time(end) + toc(t_begin)];        

    if ~store_subinfo
        train_cost = compute_cost_tensor(L_rec_all, Omega, A_Omega, tensor_dims);
        if ~isempty(Gamma) && ~isempty(A_Gamma)
            test_cost = compute_cost_tensor(Rec, Gamma, A_Gamma, tensor_dims);
        else
            test_cost = 0;
        end            
    end
    infos.train_cost = [infos.train_cost; train_cost];
    infos.test_cost = [infos.test_cost; test_cost];        
    

    if verbose > 1
        fprintf('CP-WOPU: fnum = %03d, error = %e\n', fnum, error);
    end    

    Xsol.A = xsol{1};
    Xsol.B = xsol{2};    
    Xsol.C = xsol{3};
end




