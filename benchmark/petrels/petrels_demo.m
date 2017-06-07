% first
clc;clear all;


% First we generate the data matrix and incomplete sample vector.

% Number of rows and columns
% numr = 6000;
% numc = 10000;
numr = 500;
numc = 5000;
probSize = [numr,numc];
% Rank of the underlying matrix.
truerank = 10;

% fixed size
M = 50; 

% The left and right factors which make up our true data matrix Y.
YL = randn(numr,truerank);
YR = randn(numc,truerank);

I = zeros(M*numc,1);
% Select a random set of M entries of Y.
for it = 1:numc
    p = randperm(numr);
    I((it-1)*M+1:it*M) = p(1:M);
end
J = reshape(repmat([1:numc],M,1),numc*M,1);

% Values of Y at the locations indexed by I and J.
S = sum(YL(I,:).*YR(J,:),2);
S_noiseFree = S;
    

maxrank = truerank;%i-1;
maxCycles = 1;


lambda = 0.98;

noiseFac = 0;
noise = noiseFac*randn(size(S_noiseFree));
S = S_noiseFree + noise;



[Usg, Vsg,err_reg,sub_err] = petrels_tracking(YL,I,J,S,numr,numc,maxrank,maxCycles,lambda);


%%
% % % 
figure;
semilogy(sub_err); hold on;

grid on;
xlabel('data stream index');
ylabel('normalized subspace error');

figure;
semilogy(err_reg); hold on;

grid on;
xlabel('data stream index');
ylabel('normalized residual error');