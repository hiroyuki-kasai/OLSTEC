function [Tensor_Y_Noiseless, Tensor_Y_Noiseless_Normalized, Tensor_Y_Normalized, OmegaTensor, ...
    Matrix_Y_Noiseless, Matrix_Y_Noiseless_Normalized, Matrix_Y_Normalized, OmegaMatrix, ...
    rows, cols, total_slices, Normalize_Ratio] = generate_synthetic_tensor(tensor_dims, rank, fraction, inverse_snr, data_subtype)
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017
 
    disp('# Generating synthetic dataset ....');

    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    total_slices    = tensor_dims(3);

    if strcmp(data_subtype, 'Static')

        disp('## Static dataset ....');    

        A=randn(rows, rank);
        B=randn(cols, rank);
        C=randn(total_slices, rank);

        % Create observed tensor that follows PARAFAC model
        Tensor_Y_Noiseless = zeros(rows,cols,total_slices);
        for k=1:total_slices
            Tensor_Y_Noiseless(:,:,k)=A*diag(C(k,:))*B.';
        end

    else

        disp('## Dynamic dataset ....');

        REPEAT_NUM = 4;

        SUB_SLICE = floor(total_slices/REPEAT_NUM);

        for i=1:REPEAT_NUM
            A=randn(rows,rank);
            B=randn(cols,rank);
            C=randn(SUB_SLICE,rank);   

            % Create observed tensor that follows PARAFAC model
            sub_tensor = zeros(rows,cols,SUB_SLICE);
            for k=1:SUB_SLICE
                sub_tensor(:,:,k)=A*diag(C(k,:))*B.';
            end  

            slice_start = SUB_SLICE * (i-1) + 1;
            slice_end = SUB_SLICE * i;
            Tensor_Y_Noiseless(:,:,slice_start:slice_end) = sub_tensor;
        end

    end

    Normalize_Ratio = 1*max(max(max(Tensor_Y_Noiseless)));
    Normalize_Ratio = 1;

    %% Adding noise
%     snr_val = 35;
%     aux_var = realpow(10, snr_val / 20);
%     std_noise = std(Tensor_Y_Noiseless(:)) / aux_var;
%     noise = std_noise * randn(size(Tensor_Y_Noiseless));
%     Tensor_Y = Tensor_Y_Noiseless + noise;  

    %SNR=inf;         % To add noise or not on initial tensor. 
                     % Choose SNR=inf for a noise free model
    %Noise_tens=randn(I,J,K);
    %sigma=(10^(-SNR/20))*(norm(reshape(X,J*I,K),'fro')/norm(reshape(Noise_tens,J*I,K),'fro'));
    %X=X+sigma*Noise_tens;

    %noise = [0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
    %inverse_snr = noise(noise_level);
    Tensor_Noise = randn(size(Tensor_Y_Noiseless));
    Norm_Tensor_Y_Noiseless = norm(reshape(Tensor_Y_Noiseless, rows*cols, total_slices),'fro');
    Norm_Tensor_Noise = norm(reshape(Tensor_Noise, rows*cols, total_slices),'fro');


    Tensor_Y = Tensor_Y_Noiseless + (inverse_snr * Norm_Tensor_Y_Noiseless / Norm_Tensor_Noise) * Tensor_Noise; % entries added with noise


    Tensor_Y_Noiseless_Normalized = Tensor_Y_Noiseless * Normalize_Ratio; 
    Tensor_Y_Normalized = Tensor_Y * Normalize_Ratio;


    % Matrix 
    Matrix_Y_Noiseless = reshape(Tensor_Y_Noiseless,[rows*cols total_slices]);
    Matrix_Y = reshape(Tensor_Y,[rows*cols total_slices]);

    Matrix_Y_Noiseless_Normalized = Matrix_Y_Noiseless * Normalize_Ratio;
    Matrix_Y_Normalized = Matrix_Y * Normalize_Ratio;






    %% Generate Mask Martrix/Vector
    OmegaTensor = zeros(rows,cols,total_slices);
    OmegaMatrix = zeros(rows*cols,total_slices);
    for t=1:total_slices
        % random OBSERVATION_RATIO the frame I
        M = round(fraction * rows * cols);
        p = randperm(rows * cols);
        idx = p(1:M)';

        % Omega Matrix
        OmegaMatrix(:,t) = false;
        OmegaMatrix(idx,t) = true;
        OmegaTensor(:,:,t) = reshape(OmegaMatrix(:,t),[rows,cols]);    
    end


end


