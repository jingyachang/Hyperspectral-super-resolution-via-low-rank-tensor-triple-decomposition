function [f,g]= myfun3(z,dims,Yh,Ym,P1,P2,P3,lambda)
% Evaluate the objective function and its gradient of a basically tensor
% %  z : initial point
% %  f : function value
% %  g : gradient
% 
% % Author:
% % CUI 
%% LOAD DATA
% fprintf('Load Indian Pines...\n')
% SRI = cell2mat(struct2cell(load('Indian_pines.mat')));  % ÊýÖµ»¯
% SRI(:,:,[104:108 150:163 220]) = []; %Regions of water absorption
% P3 = spectral_deg(SRI,"LANDSAT");
% MSI = tmprod(SRI,P3,3);
% d1 = 4; d2 = 4; q = 9;
% [P1,P2] = spatial_deg(SRI, q, d1, d2);
% HSI = tmprod(tmprod(SRI,P1,1),P2,2);
% [Q,S,K]=size(SRI);
% [I,J,K]=size(HSI);
% [Q,S,T]=size(MSI);
% R=2;
% dims=[Q,S,K,R,J,I,T];

% Yh=reshape(HSI,[I,J*K]);
% Ym=reshape(MSI,[Q,S*T]);

% lambda=1;

%%
% Set parameters
Q = dims(1);  S = dims(2);  K = dims(3);
L = dims(4);  M = dims(4);  N = dims(4);

J=dims(5);I=dims(6);T=dims(7);
% z=rand(Q*M*N+N*S*L+L*M*K,1);
% if length(z) ~= Q*M*N+N*S*L+L*M*K
%     error('Size of input z (%1d) don''t match (Q+S+K)R^2 (%1d).\n' ...
%         ,length(z),Q*M*N+N*S*L+L*M*K); 
% end
%%
% Abstract A,B,C
A = reshape(z(1:Q*M*N),[Q,M*N]);
B = reshape(z(Q*M*N+1:Q*M*N+N*S*L),[N*S,L]);
C = reshape(z(Q*M*N+N*S*L+1:Q*M*N+N*S*L+L*M*K),[L,M*K]);
%% gA
D=P2*reshape(B,[S,L*N]);%J*LN
% D1=kron(eye(M),reshape(D,[N,J*L]));
% D2=kron(reshape(C,[L*M,K]),eye(J));
% whos
X1=kron(eye(M),reshape(D,[N,J*L]))*kron(reshape(C,[L*M,K]),eye(J));
X=P1*A*X1-Yh;
G1=kron(eye(M),reshape(B,[N,S*L]))*kron((P3*reshape(C,[K,L*M]))',eye(S));
G=A*G1-Ym;
f=lambda*norm(X,'fro')^2+norm(G,'fro')^2;
%gA=lambda*P1'*X*X1'+ G*G1';
gA=2*lambda*((-P1'*Yh*X1')+P1'*P1*A*(X1*X1')+(A*(G1*G1')-Ym*G1'));

%% gB
Y1=kron(eye(N),C)*kron((P1*A)',eye(K));
Y=P2*reshape(B,[S,L*N])*Y1-reshape(Yh,[J,I*K]);
%f=norm(Y,'fro')^2;
E2=reshape((P3*reshape(C,[K,L*M])),[L,T*M]);
E1=kron(eye(N),E2)*kron(A',eye(T));
E=reshape(B,[S,L*N])*E1-reshape(Ym,[S,Q*T]);
% f2=norm(Y,'fro')^2+norm(E,'fro')^2;
%gB=lambda*P2'*Y*Y1'+ E*E1';
gB=2*lambda*((-P2'*reshape(Yh,[J,I*K])*Y1')+P2'*P2*reshape(B,[S,L*N])*(Y1*Y1')+(reshape(B,[S,L*N])*(E1*E1')-reshape(Ym,[S,Q*T])*E1'));


%% gC
% H3=P2*reshape(B,[S,N*L]);
% H1=kron(eye(L),reshape(P1*A,[M,I*N]))*kron(H3',eye(I));
% H=reshape(C,[K,M*L])*H1-reshape(Yh,[K,I*J]);
% 
% K1=kron(eye(L),reshape(A,[M,Q*N]))*kron(reshape(B,[N*L,S]),eye(Q));
% K=P3*reshape(C,[K,M*L])*K1-reshape(Ym,[T,Q*S]);
% % f3=norm(K,'fro')^2+norm(H,'fro')^2;
% gC=P3'*K*K1'+ lambda*H*H1';

H3=reshape((P2*reshape(B,[S,L*N])),[L,J*N]);%L*JN
H1=kron(eye(M),H3)*kron((P1*A)',eye(J));%ML*IJ
C1=reshape(C,[K,L*M]);
Yh1=reshape(Yh,[K,J*I]);
H=C1*H1-Yh1;%K*JI
Ym1=reshape(Ym,[T,Q*S]);

K1=kron(eye(M),reshape(B,[L,S*N]))*kron(reshape(A,[N*M,Q]),eye(S));%ML*QS
K=P3*C1*K1-Ym1; %T*QS
%f3=norm(K,'fro')^2+norm(H,'fro')^2;
%gC=P3'*K*K1'+ lambda*H*H1';
%%
gC=2*lambda*(P3'*P3*C1*(K1*K1')-P3'*Ym1*K1'+C1*(H1*H1')-Yh1*H1');
g=[gA(:); gB(:); gC(:)];
gnorm=norm(g);
%%
%f= f+lambda*norm(z)^2;
% g= 2*([gA(:); gB(:); gC(:)]+lambda*z);
% gNorm=norm(g,inf);
% g=2*([gA(:); gB(:); gC(:)]);
end
