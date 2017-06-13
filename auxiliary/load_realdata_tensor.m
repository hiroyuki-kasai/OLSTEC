function [Tensor_Y_Noiseless, Tensor_Y_Noiseless_Normalized, Tensor_Y_Normalized, OmegaTensor, ...
    Matrix_Y_Noiseless, Matrix_Y_Noiseless_Normalized, Matrix_Y_Normalized, OmegaMatrix, ...
    rows, cols, total_slices, Normalize_Ratio] = load_realdata_tensor(file_epath, tensor_dims, fraction)
% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 10, 2017
 
    disp('# Loading real-world dataset ....');

    rows            = tensor_dims(1);
    cols            = tensor_dims(2);
    total_slices    = tensor_dims(3);

  
    % load dataset
    X0 = importdata(file_epath);
    [m, n] = size(X0);
    
    % check image size
    if m ~= rows * cols
        fprintf('loaded image size (%d) is different from user definition (%d%d)\n', m, rows, cols);
        return;
    end
    
    % check tensor length
    if n < total_slices
        fprintf('Total slice length (%d) of loaded image size is smaller than user definition (%d).\n', n, total_slices);
        fprintf('Therefore, total slice length (%d) is changed to the real lengths (%d).\n', total_slices, n);
        total_slices = n;
    elseif n > total_slices
        fprintf('Total slice length (%d) of loaded image size is larger than user definition (%d).\n', n, total_slices);
        fprintf('Therefore, total slice length (%d) is changed to user definition (%d).\n', n, total_slices);     
        
        % reduce tensor (matrix) size
        X0 = X0(:,1:total_slices);
        [m, n] = size(X0);
    end


    Matrix_Y_Noiseless = X0;
    Tensor_Y_Noiseless = reshape(Matrix_Y_Noiseless,[rows cols total_slices]);  
    Normalize_Ratio = 1/max(max(Matrix_Y_Noiseless));

    %
    Matrix_Y_Noiseless_Normalized = Matrix_Y_Noiseless * Normalize_Ratio;
    Tensor_Y_Noiseless_Normalized = Tensor_Y_Noiseless * Normalize_Ratio; 

    %
    Matrix_Y_Normalized = Matrix_Y_Noiseless_Normalized;
    Tensor_Y_Normalized = Tensor_Y_Noiseless_Normalized;      
    


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


