function [] = test_comparison_synthetic()
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017

    clc;
    clear;
    close all;
    
    %rng('default')
    
    % set paramters
    tensor_dims = [100, 100, 400];
    tolcost     = 1e-8;
    rank        = 5;
    fraction    = 0.1;
    permute_on  = false;
    maxepochs   = 1;
    verbose     = 2;
    inverse_snr = 1e-4;

    % generate tensor (and equivalent matrix)
    %data_subtype = 'Dynamic';
    data_subtype = 'Static';
    [Tensor_Y_Noiseless, Tensor_Y_Noiseless_Normalized, Tensor_Y_Normalized, OmegaTensor, ...
        Matrix_Y_Noiseless, Matrix_Y_Noiseless_Normalized, Matrix_Y_Normalized, OmegaMatrix, ...
        rows, cols, total_slices, Normalize_Ratio] = generate_synthetic_tensor(tensor_dims, rank, fraction, inverse_snr, data_subtype);

    tensor_dims(1) = rows;
    tensor_dims(2) = cols;
    tensor_dims(3) = total_slices;
    
    numr = tensor_dims(1) * tensor_dims(2);
    numc = tensor_dims(3);    
    
    % generate init data
    Xinit.A = randn(tensor_dims(1), rank);
    Xinit.B = randn(tensor_dims(2), rank);    
    Xinit.C = randn(tensor_dims(3), rank); 
    
    
    %% CPOPT
    clear options;   
    options.maxepochs       = maxepochs*5;
    options.display_iters   = 1;
    options.store_subinfo   = true;     
    options.store_matrix    = false; 
    options.verbose         = verbose; 
    
    tic;
    [Xsol_cp_wopt, info_cp_wopt, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank, Xinit, options);
    elapsed_time_cpwopt = toc;


    %% TeCPSGD
    clear options;   
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.lambda          = 0.001;
    options.stepsize        = 0.1;
    options.mu              = 0.05;
    options.permute_on      = permute_on;    
    options.store_subinfo   = true;     
    options.store_matrix    = false; 
    options.verbose         = verbose; 

    tic; 
    [Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank, Xinit, options);
    elapsed_time_tecpsgd = toc;

    
    %% Petrels parameters
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.rank                = rank;
    options.store_subinfo       = true;     
    options.store_matrix        = false; 
    options.verbose             = verbose;
    options.lambda              = 0.80;
    
    tic; 
    [Xsol_petrels, infos_petrels, sub_infos_petrels, ~] = petrels_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);    
    elapsed_time_petrels = toc;    

    
    %% GRASTA parameters
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.permute_on          = permute_on;    
    options.verbose             = verbose;
    options.store_subinfo       = true;     
    options.store_matrix        = false; 
    options.RANK                = rank;  % the estimated rank
    options.rho                 = 1.8;    
    options.MAX_MU              = 10000; % set max_mu large enough for initial subspace training
    options.MIN_MU              = 1;
    options.ITER_MAX            = 20; 
    options.DIM_M               = rows * cols;  % your data's dimension
    options.USE_MEX             = 0;     % If you do not have the mex-version of Alg 2
                                         % please set Use_mex = 0.                                     
    tic; 
    [Xsol_grasta, infos_grasta, sub_infos_grasta, ~] = grasta_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);
    elapsed_time_grasta = toc;

    
    %% Grouse
    clear options;    
    options.maxrank         = rank;
    options.step_size       = 0.1;
    options.maxepochs       = maxepochs;       
    options.tolcost         = tolcost;
    options.permute_on      = permute_on;    
    options.store_subinfo   = true;     
    options.store_matrix    = false; 
    options.verbose         = verbose;   

    tic;        
    [Xsol_grouse, infos_grouse, sub_infos_grouse, ~] = grouse_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);
    elapsed_time_grouse = toc;
    
    
    %% OLSTEC
    clear options;
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.permute_on      = permute_on;    
    options.lambda          = 0.7;  % Forgetting paramter
    options.mu              = 0.1;  % Regualization paramter
    options.tw_flag         = 0;    % 0:Exponential Window, 1:Truncated Window (TW)
    options.tw_len          = 10;   % Window length for Truncated Window (TW) algorithm
    options.store_subinfo   = true;     
    options.store_matrix    = false; 
    options.verbose         = verbose; 

    tic;
    [Xsol_olstec, infos_olstec, sub_infos_olstec] = olstec(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank, Xinit, options);
    elapsed_time_olstec = toc;
     
    
    
    %% plotting
    fs = 20;
    figure;
    hold on;
    semilogy(sub_infos_cp_wopt.inner_iter, sub_infos_cp_wopt.err_residual, '-k', 'linewidth', 2.0);
    semilogy(sub_infos_grouse.inner_iter, sub_infos_grouse.err_residual, '-g', 'linewidth', 2.0);
    semilogy(sub_infos_grasta.inner_iter, sub_infos_grasta.err_residual, '-y', 'linewidth', 2.0);
    semilogy(sub_infos_petrels.inner_iter, sub_infos_petrels.err_residual, '-m', 'linewidth', 2.0);
    semilogy(sub_infos_TeCPSGD.inner_iter, sub_infos_TeCPSGD.err_residual, '-b', 'linewidth', 2.0);
    semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_residual, '-r', 'linewidth', 2.0);
    hold off;
    grid on;
    legend('CP-WOPT (batch)', 'Grouse (Matrix)', 'Grasta (Matrix)', 'Petrels (Matrix)', 'TeCPSGD', 'OLSTEC', 'location', 'best');
    %legend('TeCPSGD', 'OLSTEC');
    ax1 = gca;
    set(ax1,'FontSize',fs);    
    xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel('normalized residual error','FontName','Arial','FontSize',fs,'FontWeight','bold');    
    
    
    figure;
    hold on;
    semilogy(sub_infos_grouse.inner_iter, sub_infos_grouse.err_run_ave, '-g', 'linewidth', 2.0);
    semilogy(sub_infos_grasta.inner_iter, sub_infos_grasta.err_run_ave, '-y', 'linewidth', 2.0);
    semilogy(sub_infos_petrels.inner_iter, sub_infos_petrels.err_run_ave, '-m', 'linewidth', 2.0);
    semilogy(sub_infos_TeCPSGD.inner_iter, sub_infos_TeCPSGD.err_run_ave, '-b', 'linewidth', 2.0);
    semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_run_ave, '-r', 'linewidth', 2.0);
    hold off;
    grid on;
    legend('Grouse (Matrix)', 'Grasta (Matrix)', 'Petrels (Matrix)', 'TeCPSGD', 'OLSTEC', 'location', 'best');
    %legend('TeCPSGD', 'OLSTEC');
    ax1 = gca;
    set(ax1,'FontSize',fs);    
    xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
    ylabel('running average error','FontName','Arial','FontSize',fs,'FontWeight','bold');   
    
 
    fprintf('CP-WOPT:\t %.2f [sec]\n', elapsed_time_cpwopt);
    fprintf('TeCPSGD:\t %.2f [sec]\n', elapsed_time_tecpsgd);   
    fprintf('Petrels:\t %.2f [sec]\n', elapsed_time_petrels);
    fprintf('Grouse:\t\t %.2f [sec]\n', elapsed_time_grouse); 
    fprintf('Grasta:\t\t %.2f [sec]\n', elapsed_time_grasta);
    fprintf('OLSTEC:\t\t %.2f [sec]\n', elapsed_time_olstec);     
 end












