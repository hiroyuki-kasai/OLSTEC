% This file is part of OLSTEC package.
%
% Created by H.Kasai on June 07, 2017

clc;
clear;
close all;


%% set paramters
tensor_dims = [100, 100, 200];
rank        = 5;
fraction    = 0.1;
inverse_snr = 1e-4;

%% generate tensor
data_subtype = 'Static';
[A, ~, ~, Omega, ~, ~, ~, ~, ~, ~, ~, ~] = generate_synthetic_tensor(tensor_dims, rank, fraction, inverse_snr, data_subtype);


%% OLSTEC
tic;
options.verbose = 2;
[Xsol_olstec, infos_olstec, sub_infos_olstec] = olstec(A, Omega, [], tensor_dims, rank, [], options);
elapsed_time_olstec = toc;


%% plotting
fs = 20;
figure;
semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_residual, '-r', 'linewidth', 2.0);
legend('OLSTEC');
ax1 = gca;
grid on;
set(ax1,'FontSize',fs); 
xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
ylabel('normalized residual error','FontName','Arial','FontSize',fs,'FontWeight','bold');    


figure;
semilogy(sub_infos_olstec.inner_iter, sub_infos_olstec.err_run_ave, '-r', 'linewidth', 2.0);
legend('OLSTEC');
ax1 = gca;
grid on;
set(ax1,'FontSize',fs);    
xlabel('data stream index','FontName','Arial','FontSize',fs,'FontWeight','bold');
ylabel('running average error','FontName','Arial','FontSize',fs,'FontWeight','bold');   















