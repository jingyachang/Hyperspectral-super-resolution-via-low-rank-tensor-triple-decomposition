function [f,g]= myfun(z,dims,Yh,Ym,P1,P2,P3,sigma,mu,lambda)
% Evaluate the objective function and its gradient of a basically tensor
% %  z : initial point
% %  f : function value
% %  g : gradient
%%
% Set parameters
Q = dims(1);  S = dims(2);  K = dims(3);
L = dims(4);  M = dims(4);  N = dims(4);
J=dims(5);I=dims(6);T=dims(7);
%%
% Abstract A,B,C
A = reshape(z(1:Q*M*N),[Q,M*N]);
B = reshape(z(Q*M*N+1:Q*M*N+N*S*L),[N*S,L]);
C = reshape(z(Q*M*N+N*S*L+1:Q*M*N+N*S*L+L*M*K),[L,M*K]);
%% gA
D=P2*reshape(B,[S,L*N]);%J*LN
X1=kron(eye(M),reshape(D,[N,J*L]))*kron(reshape(C,[L*M,K]),eye(J));
X=P1*A*X1-Yh;
G1=kron(eye(M),reshape(B,[N,S*L]))*kron((P3*reshape(C,[K,L*M]))',eye(S));
G=A*G1-Ym;
f=norm(X,'fro')^2+lambda*norm(G,'fro')^2+mu*(norm(A,'fro')^2+norm(B,'fro')^2+norm(C,'fro')^2);
gA=2*(sigma*((-P1'*Yh*X1')+P1'*P1*A*(X1*X1'))+lambda*(A*(G1*G1')-Ym*G1')+mu*A);

%% gB
Y1=kron(eye(N),C)*kron((P1*A)',eye(K));
Y=P2*reshape(B,[S,L*N])*Y1-reshape(Yh,[J,I*K]);
E2=reshape((P3*reshape(C,[K,L*M])),[L,T*M]);
E1=kron(eye(N),E2)*kron(A',eye(T));
E=reshape(B,[S,L*N])*E1-reshape(Ym,[S,Q*T]);
gB=2*(sigma*((-P2'*reshape(Yh,[J,I*K])*Y1')+P2'*P2*reshape(B,[S,L*N])*(Y1*Y1'))+lambda*(reshape(B,[S,L*N])*(E1*E1')-reshape(Ym,[S,Q*T])*E1')+mu*reshape(B,[S,L*N]));
%% gC
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
gC=2*(lambda*(P3'*P3*C1*(K1*K1')-P3'*Ym1*K1'+sigma*(C1*(H1*H1')-Yh1*H1')+mu*C1));
g=[gA(:); gB(:); gC(:)];
%gnorm=norm(g);
%%
% f= f+lambda*norm(z)^2;
% g= 2*([gA(:); gB(:); gC(:)]+lambda*z);
% gNorm=norm(g,inf);
% g=2*([gA(:); gB(:); gC(:)]);
end
