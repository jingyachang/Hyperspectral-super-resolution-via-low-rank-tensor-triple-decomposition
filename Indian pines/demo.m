 % Author:
%   CUI XIAOFEI
%% LOAD DATA
fprintf('Load Indian_pines...\n')
SRI = cell2mat(struct2cell(load('Indian_pines.mat')));  % 数值化
SRI(:,:,[104:108 150:163 220]) = []; %Regions of water absorption
P3 = spectral_deg(SRI,"LANDSAT");
SRI(1,:,:) = []; SRI(:,1,:) = [];
% SRI=rand(10,10,16);
% P3 = spectral_deg(SRI,"Quickbird");
MSI = tmprod(SRI,P3,3);
d1 = 4; d2 = 4; q = 9;
[P1,P2] = spatial_deg(SRI, q, d1, d2);
HSI = tmprod(tmprod(SRI,P1,1),P2,2);
save  SRI P1 P2 P3 HSI MSI
%%
% subplot(1,2,1)
% imagesc(HSI(:,:,1)); title('HSI - band 1'); colorbar
% subplot(1,2,2)
% imagesc(MSI(:,:,1)); title('MSI - band 1'); colorbar
%%
[Q,S,K]=size(SRI);
[I,J,K]=size(HSI);
[Q,S,T]=size(MSI);
R=1;
dims=[Q,S,K,R,J,I,T];
Q = dims(1);  S = dims(2);  K = dims(3);
L = dims(4);  M = dims(4);  N = dims(4);
J = dims(5);  I = dims(6);  T = dims(7);
 load('z0.mat')

Yh=reshape(HSI,[I,J*K]);
Ym=reshape(MSI,[Q,S*T]);
lambda=1;

tic
options = struct('GradObj','on','Display','iter','LargeScale','on','HessUpdate','bfgs','InitialHessType','identity','GoalsExactAchieve',1,'GradConstr',true);
[x,fval2,exitflag,output,grad] = fminlbfgs(@(x)myfun3(x,dims,Yh,Ym,P1,P2,P3,lambda),z0,options);%调用函数lbfgs算法

%[x,fval2,exitflag,output,grad] = fminlbfgs(@(x)myfun6(x,dims,Yh,Ym,P1,P2,P3,F,d,lambda),z0,options);%调用函数lbfgs算法
% [x,f,k]=optLBFGS2(@myfun3,z,maxIter,memsize)
% params=struct('Display','iter','DisplayIters', 1,'MaxIters', 1000,'MaxFuncEvals', 10000,'StopTol', 1e-5,'RelFuncTol', 1e-6,'TraceX', false,'TraceFunc' ,false,'TraceRelFunc', false,'TraceGrad', false,'TraceGradNorm', false,'TraceFuncEvals', false,'LineSearch_method', 'more-thuente','LineSearch_initialstep', 1,'LineSearch_xtol', 1e-15,'LineSearch_ftol', 1e-4,'LineSearch_gtol', 1e-2,'LineSearch_stpmin', 1e-15,'LineSearch_stpmax', 1e15,'LineSearch_maxfev', 20);
% out=lbfgs(@(x)myfun3(x,dims,Yh,Ym,P1,P2,P3,lambda),z0,params);

toc
t1=toc;
disp(['运行时间',num2str(toc)])

save('x.mat')
A1 = reshape(x(1:Q*M*N),[Q,M*N]);
B1 = reshape(x(Q*M*N+1:Q*M*N+N*S*L),[N*S,L]);
C1 = reshape(x(Q*M*N+N*S*L+1:Q*M*N+N*S*L+L*M*K),[L,M*K]);
SRI_hat1 = reshape(A1*reshape(permute(reshape(B1*C1,[N,S,M,K]),[3,1,2,4]),[M*N,S*K]),[Q,S,K]);
%SRI_hat1= reshape(X1,[Q,S,K]);
%%
% figure(2)
% subplot(1,2,1)
% imagesc(SRI(:,:,44)); title('Groundtruth SRI - band 44')
% subplot(1,2,2)
% imagesc(SRI_hat1(:,:,44)); title('Result of TTDSR - band 44')
%%
d1=4;d2=4;
err1 = cell2mat(compute_metrics(SRI,SRI_hat1,d1,d2));
tab = ["TTD Algorithm" "R-SNR" "CC" "SAM" "ERGAS" "TIMES";...
       "Indian_pines" err1 toc];
   fprintf('Plot comparison metrics \n')   
tab

