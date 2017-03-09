function [x,b,u,w,C,res_p,res_d] = rhuwacb_admm(A,y,wv,delta,varargin)
% [x,b,res_p,res_d] = rhuwacb_admm(A,y,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% alternating direction method of multipliers (ADMM)
%
%  Inputs
%     A : dictionary matrix (L x Na) where Na is the number of atoms in the
%     library and L is the number of wavelength bands
%     y : observation vector (L x N) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
%     delta
%  Optional parameters
%     Tol: tolearance
%     Maxiter: maximum number of iterations
%  Outputs
%     X: estimated abundances (Na x N)
%     B: estimated concave background (L x N)
%     res_p,res_d: primal and dual residuals
%
%--------------------------------------------------------------------------
%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||Y-AX-B||^2_F
%           X,B
%         subject to  X>=0 and CB>=delta
%  where C is a concave operator that ensures the concavity of the
%  multiplied vector.
%
%

%%
%--------------------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------------------
if (nargin-length(varargin)) ~= 3
    error('Wrong number of required parameters');
end
% mixing matrixsize
[LA,N] = size(A);
% data set size
[L,Ny] = size(y);
if (LA ~= L)
    error('mixing matrix M and data set y are inconsistent');
end

wv = wv(:);
Lwv = length(wv);
if (L~=Lwv)
    error('the wavelength samples wv is not correct.');
end

if length(delta)==0
    error('delta needs to be a scaler or the size-2 of the wavelength');
elseif length(delta)>1
    if length(delta) ~= (length(wv)-2)
        error('delta is invalid');
    end
end
    

%%
%--------------------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------------------
% maximum number of AL iteration
maxiter = 5000;
% display only sunsal warnings
verbose = 'off';
% tolerance for the primal and dual residues
tol = 1e-4;
% initialization of X0
x0 = 0;
% initialization of B0
b0 = 0;

%%
%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAXITER'
                maxiter = round(varargin{i+1});
                if (maxiter <= 0 )
                       error('AL_iters must a positive integer');
                end
            case 'TOL'
                tol = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
%             case 'X0'
%                 x0 = varargin{i+1};
%                 if (size(x0,1) ~= p) | (size(x0,1) ~= N)
%                     error('initial X is  inconsistent with M or Y');
%                 end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end


% compute mean norm
ynrm = vnorms(y,1);
% rescale M and Y and lambda
% A = A/ynrm;
% y = y/ynrm;

%%
%--------------------------------------------------------------------------
% Create the concave operator
%--------------------------------------------------------------------------
C = concaveOperator(wv);

%%
%--------------------------------------------------------------------------
% pre-processing for main loop
%--------------------------------------------------------------------------
rho = 0.01;
Cinv = C\eye(L);
s_c = vnorms(Cinv,1);
Cinv = bsxfun(@rdivide,Cinv,s_c);
C = bsxfun(@times,C,s_c');
if isempty(A)
    T = [Cinv];
else
    T = [A Cinv];
end
[V,Sigma] = svd(T'*T);
V(N+1:N+L,:) = Cinv * V(N+1:N+L,:);
Sigma = diag(Sigma);
Sigmarhoinv = 1./(Sigma + rho);
Q = bsxfun(@times,V,Sigmarhoinv') * V.';
ay = A.' * y;

%%
%--------------------------------------------------------------------------
% Initialization
%--------------------------------------------------------------------------
if x0 == 0
    s= Q*[ay;y];
    x = s(1:N,:);
    x(x<0) = 0;
end
b = zeros([L,Ny]);
% augmented variables
u = x;
w = C*b;
% w(2:L-1,:) = max(w(2:L-1,:),0);
% dual variables
du = zeros([N,Ny]); dw = zeros([L,Ny]);

%%
%--------------------------------------------------------------------------
% main loop
%--------------------------------------------------------------------------
tol_p = sqrt((L+N)*Ny)*tol;
tol_d = sqrt((L+N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
change_rho = 0;
% Cdiag = diag(C); Cdiagu = diag(C(:,2:end)); Cdiagl=diag(C(2:end,:));

while (k <= maxiter) && ((abs (res_p) > tol_p) || (abs (res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 1
        u0 = u; w0 = w;
    end
    % update X and B
    s = Q * [ay + rho*(u-du);y + rho*C.'*(w-dw)];
%     wdw = w-dw;
%     tmp = bsxfun(@times,wdw,Cdiag);
%     tmp(1:end-1,:) = tmp(1:end-1,:) + bsxfun(@times,wdw(2:end,:),Cdiagl);
%     tmp(2:end,:) = tmp(2:end,:) + bsxfun(@times,wdw(1:end-1,:),Cdiagu);
%     s = Q * [ay + rho*(u-du);y + rho*tmp];
    x = s(1:N,:);
    b = s(N+1:N+L,:);
    
    % update u and w
    u = max(x+du,0);
%     tmp = bsxfun(@times,b,Cdiag);
%     tmp(1:end-1,:) = tmp(1:end-1,:) + bsxfun(@times,b(2:end,:),Cdiagu);
%     tmp(2:end,:) = tmp(2:end,:) + bsxfun(@times,b(1:end-1,:),Cdiagl);
    tmp2 = C*b;
    w = tmp2 + dw;
%     w = C*b + dw;
    w(2:L-1,:) = max(w(2:L-1,:),delta);
    
    % update the dual variables
    xminusu = x-u;
    du = du + xminusu;
    Cbminusw = tmp2-w;
    dw = dw + Cbminusw;

    % update mu so to keep primal and dual residuals whithin a factor of 10
    if mod(k,10) == 1
        % primal residue
        res_p = norm(xminusu,'fro') + norm(Cbminusw,'fro');
        % dual residue
%         ww0 = w-w0;
%         tmp2 = bsxfun(@times,ww0,Cdiag);
%         tmp2(1:end-1,:) = tmp2(1:end-1,:) + bsxfun(@times,ww0(2:end,:),Cdiagl);
%         tmp2(2:end,:) = tmp2(2:end,:) + bsxfun(@times,ww0(1:end-1,:),Cdiagu);
%         res_d = rho*(norm(u-u0,'fro') + norm(tmp2,'fro'));
        res_d = rho*(norm(u-u0,'fro') + norm(C.'*(w-w0),'fro'));
        if  strcmp(verbose,'yes')
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
        % update mu
        if res_p > 10*res_d
            rho = rho*2;
            du = du/2; dw = dw/2;
            change_rho = 1;
        elseif res_d > 10*res_p
            rho = rho/2;
            du = du*2; dw = dw*2;
            change_rho = 1;
        end
        if  change_rho
            % Px and Pb
            Sigmarhoinv = 1./(Sigma + rho);
            Q = bsxfun(@times,V,Sigmarhoinv') * V.';
            change_rho = 0;
        end
    end
    k=k+1;    
end
