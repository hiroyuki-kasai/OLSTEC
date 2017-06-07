%clear all; clc; close all;

X = importdata('CV/hall1-200.mat');

%% predefine your video parameters

total_fnum = size(X,2);

VIDEO_ROWS  = 144;
VIDEO_COLS  = 176;
DIM         = VIDEO_ROWS * VIDEO_COLS;

%% GRASTA parameters
subsampling                 = 0.1; % how much patial information will be used in your application
OPTIONS.RANK                = 10;  % the estimated rank
OPTIONS.rho                 = 1.8;    
OPTIONS.MAX_MU              = 10000; % set max_mu large enough for initial subspace training
OPTIONS.MIN_MU              = 1;
OPTIONS.ITER_MAX            = 20; 
OPTIONS.DIM_M               = DIM;  % your data's dimension
OPTIONS.USE_MEX             = 1;     % If you do not have the mex-version of Alg 2

                                     % please set Use_mex = 0.                                     

OPTS                        = struct(); % initiate a empty struct for OPTS

status.init                 = 0;        % status of grasta at each iteration

U_hat                       = zeros(1); % initiate a zero U_hat at beginning 

%% Initial subspace training [optional]

% You may use some frames to train the initial subspace. 

if 
    OPTIONS.CONSTANT_STEP       = 0; % use adaptive step-size for initial subspace training
    max_cycles                  = 10;
    training_frames             = 100;

    for outiter = 1:max_cycles,

        frame_order = randperm(training_frames);

        for i=1:training_frames,       

            % prepare the training frame
            %I = imread(fname);
            I = X(:,frame_order(i));

            % I = double(rgb2gray(I));        
            I = I/max(max(I));

            % random subsampling the frame I
            M = round(subsampling * DIM);
            p = randperm(DIM);
            idx = p(1:M)';

            I_Omega = I(idx);       

            % training the background        
            [U_hat, status, OPTS] = grasta_stream(I_Omega, idx, U_hat, status, OPTIONS, OPTS);        

        end

        fprintf('Training %d/%d ...\n',outiter, max_cycles);

    end
else
    U_hat                       = orth(randn(dim,rank));
    status.w                    = U\D(:,1);
    status.SCALE                = 1;
end

bg_img_init = reshape(U_hat * status.w * status.SCALE, VIDEO_ROWS,VIDEO_COLS);

%% Real-time background/foreground separation

OPTIONS.CONSTANT_STEP       = 1e-2; % use small constant step-size

figure;
for fnum=1:total_fnum
    % prepare the image I whether it is saved as file or caputured by
    % camera
    % I = imread(fname); 
    I = X(:,fnum);
    
    %I = double(rgb2gray(I));        
    I = I/max(max(I));

    % random subsampling the frame I
    M = round(subsampling * DIM);
    p = randperm(DIM);
    idx = p(1:M)';

    I_Omega = I(idx);
    o_img = reshape(I, VIDEO_ROWS,VIDEO_COLS);


    % tracking the background
    [U_hat, status, OPTS] = grasta_stream(I_Omega, idx, U_hat, status, OPTIONS, OPTS);  

    % bg_img is the background
    bg_img = reshape(U_hat * status.w * status.SCALE, VIDEO_ROWS,VIDEO_COLS);
    

    % s_img is the separated foreground
    s_hat = I(:) - U_hat * status.w * status.SCALE;
    s_img = reshape(s_hat,VIDEO_ROWS,VIDEO_COLS); 
    

    subplot(1,4,1);imagesc(o_img);colormap(gray);axis image;axis off;title('Input');
    subplot(1,4,2);imagesc(bg_img_init);colormap(gray);axis image;axis off;title('Back (Initial)');
    subplot(1,4,3);imagesc(bg_img);colormap(gray);axis image;axis off;title(['Back(L), f=', num2str(fnum)]);
    subplot(1,4,4);imagesc(s_img);colormap(gray);axis image;axis off;title('Foreground');   
    pause(0.1);

     
    fprintf('processing: %d\n', fnum);
end





disp('finish!');