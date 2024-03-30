%%非凸优化问题找合适的初值
load('ST')
[I,J,K] = size(ST);
R = 1; %秩r的选择
S1=reshape(ST,[I,K*J])';
%R=1:3
FitErr = -ones(length(R),2);
nmse=-ones(length(R),2);
rsnr=-ones(length(R),2);
for j=1:length(R)
    %R1 = 5*R(j);  R2 = 4*R(j);  R3 = R(j);
    R1=40; R2=40; R3=4;
    fprintf('\nLow-Rank Approximation %dx%dx%d ... ...\n',R1,R2,R3);
    tic;
    [TD,U,fe] = TriDB(ST,[R1,R2,R3,R(j)]);%triple decomposition
    D = reshape(permute(reshape(reshape(TD{1},[R1*R(j),R(j)])*reshape(TD{2},[R(j)*R2,R(j)])',[R1,R(j),R(j),R2]),[1,4,3,2]),[R1*R2,R(j)*R(j)]) * reshape(TD{3},[R(j)*R(j),R3]);
    D = U{1}*reshape(reshape(reshape(D',[R3*R1,R2])*U{2}',[R3,R1*J])'*U{3}',[R1,J*K]);
    DD=reshape(D,[I,J,K]);
    nmse=norm(S1-D','fro')/norm(S1,'fro'); 
    rsnr=10*log10(norm(S1,'fro')^2/norm(S1-D','fro')^2); 
    nmse(j,2)=nmse;
    rsnr(j,2)=rsnr;
    FitErr(j,2) = fe;
end
tic
A0=tmprod(TD{1},U{1},1);
B0=tmprod(TD{2},U{2},2);
C0=tmprod(TD{3},U{3},3);
[Q,M,N]=size(A0);
[L,S,N]=size(B0);
[L,M,G]=size(C0);
A1 = reshape(A0,[Q,M*N]);
B1 = reshape(B0,[L,N*S])';
C1 = reshape(C0,[L,M*G]);
ZZ0=[tens2mat(A0,1)' tens2mat(B0,2)' tens2mat(C0,3)']';%向量化
z0=ZZ0(:);%向量化
GG = reshape(A1*reshape(permute(reshape(B1*C1,[N,S,M,G]),[3,1,2,4]),[M*N,S*G]),[Q,S,G]);
GG1=tens2mat(GG,1)';
nmse3=norm(S1-GG1,'fro')/norm(S1,'fro');
rsnr3=10*log10(norm(S1,'fro')^2/norm(S1-GG1,'fro')^2);
toc
disp(['运行时间:',num2str(toc)])