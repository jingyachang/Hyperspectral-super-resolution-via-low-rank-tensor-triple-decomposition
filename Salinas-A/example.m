%% LOAD DATA
clear clc 
%%
% SRI = cell2mat(struct2cell(load('Indian_pines.mat')));
% SRI(:,:,[104:108 150:163 220]) = []; %Regions of water absorption
% SRI(1,:,:) = []; SRI(:,1,:) = [];
% P3 = spectral_deg(SRI,"LANDSAT");
% MSI = tmprod(SRI,P3,3);
% d1 = 6; d2 = 6; q = 9;
% [P1,P2] = spatial_deg(SRI, q, d1, d2);
% HSI = tmprod(tmprod(SRI,P1,1),P2,2);
%%
% SRI = cell2mat(struct2cell(load('SalinasA.mat')));
% SRI(:,:,[108:112 154:167 224]) = []; %Regions of water absorption (Salinas)
% SRI = crop(SRI,[80,84,size(SRI,3)]);
% P3 = spectral_deg(SRI,"LANDSAT");
% d1 = 4; d2 = 4; q = 9;
% [P1,P2] = spatial_deg(SRI, q, d1, d2);
% MSI = tmprod(SRI,P3,3); HSI = tmprod(tmprod(SRI,P1,1),P2,2);
%% load data
SRI = cell2mat(struct2cell(load('ST.mat')));
%% degradation
P3 = spectral_deg(SRI,"Quickbird");
d1 = 4; d2 = 4; q = 9;%d1 = 6; d2 = 6; q = 9;
[P1,P2] = spatial_deg(SRI, q, d1, d2);
MSI = tmprod(SRI,P3,3); HSI = tmprod(tmprod(SRI,P1,1),P2,2);
for k=1:size(HSI,3)
    HSI(:,:,k) = awgn(HSI(:,:,k),30,'measured');
end
for k=1:size(MSI,3)
    MSI(:,:,k) = awgn(MSI(:,:,k),30,'measured');
end
%%
[Q,S,K]=size(SRI);
S1=reshape(SRI,[Q,K*S])';
[I,J,K]=size(HSI);
[Q,S,T]=size(MSI);
R=1;
mu=1;
sigma=1;
lambda=1;
dims=[Q,S,K,R,J,I,T];
Q = dims(1);  S = dims(2);  K = dims(3);
L = dims(4);  M = dims(4);  N = dims(4);
J = dims(5);  I = dims(6);  T = dims(7);
load('z0.mat')
%z0=ones(Q*M*N+L*S*N+L*M*K,1);
Yh=tens2mat(HSI,1);
Ym=tens2mat(MSI,1);
%%
tic
options = struct('GradObj','on','Display','iter','LargeScale','on','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',true);
[x,fval2,exitflag,output,grad] = fminlbfgs(@(x)myfun(z0,dims,Yh,Ym,P1,P2,P3,sigma,mu,lambda),z0,options);%调用lbfgs算法
disp(['运行时间:',num2str(toc)])
A1 = reshape(x(1:Q*M*N),[Q,M*N]);
B1 = reshape(x(Q*M*N+1:Q*M*N+N*S*L),[N*S,L]);
C1 = reshape(x(Q*M*N+N*S*L+1:Q*M*N+N*S*L+L*M*K),[L,M*K]);
SRI_hat2 = reshape(A1*reshape(permute(reshape(B1*C1,[N,S,M,K]),[3,1,2,4]),[M*N,S*K]),[Q,S,K]);
toc
%%
S1_hat2=tens2mat(SRI_hat2,1)';
nmse2=norm(S1-S1_hat2,'fro')/norm(S1,'fro'); %NMSE
RRsnr=10*log10(norm(S1,'fro')^2/norm(S1-S1_hat2,'fro')^2); %RSNR
err = cell2mat(compute_metrics(SRI,SRI_hat2,d1,d2));
tab = ["TTDSR" "R-SNR" "CC" "SAM" "ERGAS" "TIMES";...
       "Salinas-A" err toc];
   fprintf('Plot comparison metrics \n')   
tab