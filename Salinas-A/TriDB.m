function [trTen,UVW,FE] = TriDB(X,rdims)
% Compute Low-Rank Triple Decomposition of a thrid order tensor.
%
% Input:
%     X         Third order tensor
%     rdims     [r_1,r_2,r_3,rr]
%
% OutPut:
%     Ah,Bh,Ch  Core -- Triple Tensor
%     U,V,W     Tucker factors
%     FE        Relative fitting error
%
% Initialization
[sz,I] = sort(size(X)); 
if length(sz)~=3, error('Input tensor must be of third order!'), end
X = permute(X,I);  cyn = max(abs(X(:)));  X = X/cyn;
N1 = sz(1);       N2 = sz(2);       N3 = sz(3);
R1 = rdims(I(1)); R2 = rdims(I(2)); R3 = rdims(I(3));
r1 = rdims(4);    r2 = rdims(4);    r3 = rdims(4);

X1 = reshape(X,[N1,N2*N3]);  FO = norm(X1,'fro');
[U0,~,~] = svds(X1,R1);  D0 = reshape(X1'*U0,[N2,N3*R1]);
[V0,~,~] = svds(D0,R2);  D0 = reshape(D0'*V0,[N3,R1*R2]);
[W0,~,~] = svds(D0,R3);  D0 = D0'*W0;                                                          %<--D0 is of size [R1*R2,R3].

if R3>=r3, [Ch,~,~] = svds(D0,r3); else [Ch,~,~] = svds([D0,randn(R1*R2,r3-R3)],r3); end;  Ch = reshape(Ch,[R1,R2*r3]);
[~,~,Ah] = svds(reshape(Ch',[R2,r3*R1]),r2);  Ah = reshape(Ah,[r3,R1*r2]);                     %<-- Ah is of size [r3,R1*r2].
[~,~,Bh] = svds(Ch,r1);                       Bh = Bh';                                        %<-- Bh is of size [r1,R2*r3].
MAB = reshape(permute(reshape(reshape(Bh,[r1*R2,r3])*Ah,[r1,R2,R1,r2]),[3,2,1,4]),[R1*R2,r1*r2]);
Ch = MAB \ D0;  D0 = MAB*Ch;                  Ch = reshape(reshape(Ch,[r1,r2*R3])',[r2,R3*r1]);%<-- Ch is of size [r2,R3*r1].
D0 = U0*reshape(V0*reshape(reshape(D0*W0',[R1,R2*N3])',[R2,N3*R1]),[N2*N3,R1])';%[N1,N2*N3]
F0 = (norm(D0 - X1,'fro') / FO)^2;  Fh = F0+1:5;

% Main iterate
gamma   = 1.6;
lambda  = 0.001;
maxiter = 100;
UTU = U0'*U0;  VTV = V0'*V0;  WTW = W0'*W0;
for iter=1:maxiter
    fprintf('\b\b\b\b\b\b%6d',iter);
    
    % Update Ah and U ...
    NN = reshape(permute(reshape(reshape(Ch,[r2*R3,r1])*Bh,[r2,R3,R2,r3]),[1,4,3,2]),[r2*r3,R2*R3]);
    MM = reshape(reshape(reshape(reshape(NN,[r2*r3*R2,R3])',[R3*r2*r3,R2])*VTV,[R3,r2*r3*R2])'*WTW,[r2*r3,R2*R3]) * NN';  Ah = reshape(Ah',[R1,r2*r3]);
    XM = (NN * reshape((reshape((reshape(X,[N1*N2,N3])*W0)',[R3*N1,N2])*V0)',[R2*R3,N1]))';%[N1,r2*r3]
    % Ak = (kron(MM,UTU) + lambda*speye(R1*r2*r3)) \ reshape(U0'*XM + lambda*Ah,[R1*r2*r3,1]);  Ak = reshape(Ak,[R1,r2*r3]);
    Ak = SME(UTU,MM,U0'*XM+lambda*Ah,lambda);
    Uk = (XM*Ak'+lambda*U0) / (Ak*MM*Ak'+lambda*eye(R1));
    Ak = Ah + gamma*(Ak-Ah);  Uk = U0 + gamma*(Uk-U0);
    % D0 = reshape(Ak*NN,[R1*R2,R3]); D0 = Uk*reshape(V0*reshape(reshape(D0*W0',[R1,R2*N3])',[R2,N3*R1]),[N2*N3,R1])';  F0 = (norm(D0 - X1,'fro') / FO)^2;
    Ak = reshape(Ak,[R1*r2,r3])';  Ah = reshape(Ah,[R1*r2,r3])';  UTU = Uk'*Uk;
    
    % Update Bh and V ...
    NN = reshape(permute(reshape(reshape(Ak,[r3*R1,r2])*Ch,[r3,R1,R3,r1]),[1,4,3,2]),[r3*r1,R3*R1]);
    MM = reshape(reshape(reshape(reshape(NN,[r3*r1*R3,R1])',[R1*r3*r1,R3])*WTW,[R1,r3*r1*R3])'*UTU,[r3*r1,R3*R1]) * NN';  Bh = reshape(Bh',[R2,r3*r1]);
    XM = reshape(reshape(reshape(X,[N1*N2,N3])*W0,[N1,N2*R3])'*Uk,[N2,R3*R1]) * NN';%[N2,r3*r1]
    % Bk = (kron(MM,VTV) + lambda*speye(r1*R2*r3)) \ reshape(V0'*XM + lambda*Bh,[r1*R2*r3,1]);  Bk = reshape(Bk,[R2,r3*r1]);  
    Bk = SME(VTV,MM,V0'*XM+lambda*Bh,lambda);
    Vk = (XM*Bk'+lambda*V0) / (Bk*MM*Bk'+lambda*eye(R2));
    Bk = Bh + gamma*(Bk-Bh);  Vk = V0 + gamma*(Vk-V0);
    % D0 = reshape(reshape(Bk*NN,[R2*R3,R1])',[R1*R2,R3]); D0 = Uk*reshape(Vk*reshape(reshape(D0*W0',[R1,R2*N3])',[R2,N3*R1]),[N2*N3,R1])';  F0 = (norm(D0 - X1,'fro') / FO)^2;
    Bk = reshape(Bk,[R2*r3,r1])';  Bh = reshape(Bh,[R2*r3,r1])';  VTV = Vk'*Vk;
    
    % Update Ch and W ...
    NN = reshape(permute(reshape(reshape(Bk,[r1*R2,r3])*Ak,[r1,R2,R1,r2]),[1,4,3,2]),[r1*r2,R1*R2]);
    MM = reshape(reshape(reshape(reshape(NN,[r1*r2*R1,R2])',[R2*r1*r2,R1])*UTU,[R2,r1*r2*R1])'*VTV,[r1*r2,R1*R2]) * NN';  Ch = reshape(Ch',[R3,r1*r2]);
    XM = reshape(reshape(reshape(X,[N1,N2*N3])'*Uk,[N2,N3*R1])'*Vk,[N3,R1*R2]) * NN';%[N3,r1*r2]
    % Ck = (kron(MM,WTW) + lambda*speye(r1*r2*R3)) \ reshape(W0'*XM + lambda*Ch,[r1*r2*R3,1]);  Ck = reshape(Ck,[R3,r1*r2]);
    Ck = SME(WTW,MM,W0'*XM+lambda*Ch,lambda);
    Wk = (XM*Ck'+lambda*W0) / (Ck*MM*Ck'+lambda*eye(R3));
    Ck = Ch + gamma*(Ck-Ch);  Wk = W0 + gamma*(Wk-W0);  
    D0 = (Ck*NN)'; D0 = Uk*reshape(Vk*reshape(reshape(D0*Wk',[R1,R2*N3])',[R2,N3*R1]),[N2*N3,R1])';  F0 = (norm(D0 - X1,'fro') / FO)^2;
    Ck = reshape(Ck,[R3*r1,r2])';  Ch = reshape(Ch,[R3*r1,r2])';  WTW = Wk'*Wk;  Fh = [F0,Fh(1:4)];

    % Next iteration
    nX = [norm(Ak,'fro'),norm(Bk,'fro'),norm(Ck,'fro'),norm(Uk,'fro'),norm(Vk,'fro'),norm(Wk,'fro')];  %disp([iter, F0, nX]);
    dX = [norm(Ah-Ak,'fro'),norm(Bh-Bk,'fro'),norm(Ch-Ck,'fro'),norm(U0-Uk,'fro'),norm(V0-Vk,'fro'),norm(W0-Wk,'fro')];  %disp([iter, F0, dX]);
    % if norm(dX)<1.e-3*norm(nX) || abs(Fh(1)-Fh(5))<1.e-12*(1+Fh(5))
    if abs(sqrt(Fh(1))-sqrt(Fh(2)))<1.e-4
        disp([iter, F0, nX]);  disp([iter, F0, dX]);
        break;
    end
    
    Ah = Ak;  Bh = Bk;  Ch = Ck;  U0 = Uk;  V0 = Vk;  W0 = Wk;
     lambda = max(0.7*lambda,1.e-6);
    % lambda = max(0.7*lambda,1.e-6);
end

% Return data
trTen{I(1)} = ipermute(reshape(Ak',[R1,r2,r3]),I);
trTen{I(2)} = ipermute(reshape(Bk,[r1,R2,r3]),I);
trTen{I(3)} = ipermute(reshape(reshape(Ck,[r2*R3,r1])',[r1,r2,R3]),I);
UVW{I(1)} = Uk;  UVW{I(2)} = Vk;  UVW{I(3)} = Wk*cyn;
FE = sqrt(F0);
