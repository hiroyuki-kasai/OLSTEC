function [] = test_comparison_real()
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 13, 2017

    clc;
    clear;
    close all;
    
    
    % set running flags
    image_display_flag  = true;
    store_matrix_flag   = true;
    permute_on_flag     = false;
    maxepochs           = 1;
    verbose             = 2;   
    tolcost             = 1e-8;
    
    % set paramters
    rank                = 20;

    fraction            = 0.1;
    
    % set dataset
    data_type   = 'dynamic';%'static'; % 'dynamic'; 
    if strcmp(data_type, 'static')
        %file_path   =  './dataset/hall/hall_144x100_frame2900-3899.mat';
        file_path   =  './dataset/hall/hall1-200.mat';
        tensor_dims = [144, 176, 100];      
    else
        file_path   =  './dataset/hall/hall_144x100_frame2900-3899_pan.mat';
        tensor_dims = [144, 100, 500];
    end

    % load tensor (and equivalent matrix)
    [Tensor_Y_Noiseless, Tensor_Y_Noiseless_Normalized, Tensor_Y_Normalized, OmegaTensor, ...
        Matrix_Y_Noiseless, Matrix_Y_Noiseless_Normalized, Matrix_Y_Normalized, OmegaMatrix, ...
        rows, cols, total_slices, Normalize_Ratio] = load_realdata_tensor(file_path, tensor_dims, fraction);

    % revise tensor_dims
    tensor_dims(1) = rows;
    tensor_dims(2) = cols;
    tensor_dims(3) = total_slices;
    
    % set paramter for matrix case
    numr = tensor_dims(1) * tensor_dims(2);
    numc = tensor_dims(3);  
    
    % calculate matrix rank 
    num_params_of_tensor = rank * sum(tensor_dims,2);
    matrix_rank = floor( num_params_of_tensor/ (numr+numc) );
    if matrix_rank < 1
        matrix_rank = 1;
    end
    
    % generate init data
    Xinit.A = randn(tensor_dims(1), rank);
    Xinit.B = randn(tensor_dims(2), rank);    
    Xinit.C = randn(tensor_dims(3), rank); 
    
    
    %% CPOPT (batch)
    clear options;   
    options.maxepochs       = maxepochs;
    options.display_iters   = 1;
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose; 
    
    tic;
    [Xsol_cp_wopt, info_cp_wopt, sub_infos_cp_wopt] = cp_wopt_mod(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank, Xinit, options);
    elapsed_time_cpwopt = toc;

    
    %% Petrels parameters (matrix)
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.rank                = matrix_rank;
    options.permute_on          = permute_on_flag;    
    options.store_subinfo       = true;     
    options.store_matrix        = store_matrix_flag; 
    options.verbose             = verbose;
    options.lambda              = 0.98;
    
    tic; 
    [Xsol_petrels, infos_petrels, sub_infos_petrels, ~] = petrels_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);    
    elapsed_time_petrels = toc;    

    
    %% GRASTA parameters (matrix)
    clear options;
    options.maxepochs           = maxepochs;
    options.tolcost             = tolcost;
    options.permute_on          = permute_on_flag;    
    options.verbose             = verbose;
    options.store_subinfo       = true;     
    options.store_matrix        = store_matrix_flag; 
    options.RANK                = matrix_rank;
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

    
    %% Grouse (matrix)
    clear options;    
    options.maxrank         = matrix_rank;
    options.step_size       = 0.0001;
    options.maxepochs       = maxepochs;       
    options.tolcost         = tolcost;
    options.permute_on      = permute_on_flag;    
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose;   

    tic;        
    [Xsol_grouse, infos_grouse, sub_infos_grouse, ~] = grouse_mod([], Matrix_Y_Noiseless, OmegaMatrix, [], numr, numc, options);
    elapsed_time_grouse = toc;
    

    %% TeCPSGD
    clear options;   
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.lambda          = 0.001;
    options.stepsize        = 0.1;
    options.mu              = 0.05;
    options.permute_on      = permute_on_flag;    
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
    options.verbose         = verbose; 

    tic; 
    [Xsol_TeCPSGD, info_TeCPSGD, sub_infos_TeCPSGD] = TeCPSGD(Tensor_Y_Noiseless, OmegaTensor, [], tensor_dims, rank, Xinit, options);
    elapsed_time_tecpsgd = toc;    
    
    
    %% OLSTEC
    clear options;
    options.maxepochs       = maxepochs;
    options.tolcost         = tolcost;
    options.permute_on      = permute_on_flag;    
    options.lambda          = 0.7;  % Forgetting paramter
    options.mu              = 0.1;  % Regualization paramter
    options.tw_flag         = 0;    % 0:Exponential Window, 1:Truncated Window (TW)
    options.tw_len          = 10;   % Window length for Truncated Window (TW) algorithm
    options.store_subinfo   = true;     
    options.store_matrix    = store_matrix_flag; 
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
    
    
    %% Display images
    observe = 100 * (1 - fraction);
    if image_display_flag
        figure;
        width = 5;
        height = 3;
        for i=1:total_slices
             
            display_images(rows, cols, observe, height, width, 1, i, sub_infos_petrels, 'Petrels');
            display_images(rows, cols, observe, height, width, 2, i, sub_infos_grasta, 'Grasta');
            display_images(rows, cols, observe, height, width, 3, i, sub_infos_grouse, 'Grouse');
            display_images(rows, cols, observe, height, width, 4, i, sub_infos_TeCPSGD, 'TeCPSGD');
            display_images(rows, cols, observe, height, width, 5, i, sub_infos_olstec, 'OLSTEC');            

            pause(0.1);
        end
    end    
end
 

function display_images(rows, cols, observe, height, width, test, frame, sub_infos, algorithm)

        subplot(height, width, 1 + (test-1));
        imagesc(reshape(sub_infos.I(:,frame),[rows cols]));
        colormap(gray);axis image;axis off;
        title([algorithm, ': ', num2str(observe), '% missing']); 

        subplot(height, width, width + 1 + (test-1));
        imagesc(reshape(sub_infos.L(:,frame),[rows cols]));
        colormap(gray);axis image;axis off;
        title(['Low-rank image: f = ', num2str(frame)]);

        subplot(height, width, 2*width + 1 + (test-1));
        imagesc(reshape(sub_infos.E(:,frame),[rows cols]));
        colormap(gray);axis image;axis off;
        title(['Residual image: error = ', num2str(sub_infos.err_residual(frame))]);    
        
end












