function [x,z,C,rho,res_p,res_d] = huwacbl1_admm2(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacbl1_admm(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% alternating direction method of multipliers (ADMM)
% sparsity constraint on 
%
%  Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x Ny) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
%  Optional parameters
%     'TOL': tolearance (default) 1e-4
%     'MAXITER' : maximum number of iterations (default) 1000
%     'VERBOSE' : {'yes', 'no'}
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%     'X0'      : Initial x (coefficient vector/matrix for the libray A)
%                 (default) 0
%     'Z0'      : Initial z (coefficient vector/matrix for the concave
%                 bases C) (default) 0
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%     'B0'      : Initial Background vector B [L x N]. This will be converted to C
%                 (default) 0
%     'D0'      : Initial dual parameters [N+L,L] (non-scaling form)
%                 (default) 0
%     'rho'     : initial spectral penalty parameter, scalar
%                 (default) 0.01
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z
%     d: estimated dual variables (N+L x Ny)
%     rho: spectral penalty parameter at the convergence 
%     res_p,res_d: primal and dual residuals for feasibility

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    ||y-Ax-Cz||^1 + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
%
%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin-length(varargin)) ~= 3
    error('Wrong number of required parameters');
end
% mixing matrixsize
Aisempty = isempty(A);
if Aisempty
    N = 0;
else
    [LA,N] = size(A);
end
% data set size
[L,Ny] = size(y);
if ~Aisempty
    if (LA ~= L)
        error('mixing matrix M and data set y are inconsistent');
    end
end
if ~isvector(wv) || ~isnumeric(wv)
    error('wv must be a numeric vector.');
end
wv = wv(:);
Lwv = length(wv);
if (L~=Lwv)
    error('the wavelength samples wv is not correct.');
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = 1000;
% display only sunsal warnings
verbose = false;
% tolerance for the primal and dual residues
tol = 1e-4;
% weights for each dimensions
weight = ones([L,1]);
% sparsity constraint on the library
lambda_a = 0.0;
% spectral penalty parameter
% rho = 0.01;
rho = ones([1,Ny]);

% % nonnegativity constraint on the library
% nonneg = ones([N,1]);

% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
% initialization of B
b0 = 0;
% initialization of Lagrange multipliers, d0
d0 = 0;
% base matrix of concave curvature
C = 0;

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
                if strcmp(varargin{i+1},'yes')
                    verbose=true;
                elseif strcmp(varargin{i+1},'no')
                    verbose=false;
                else
                    error('verbose is invalid');
                end               
            case 'WEIGHT'
                weight = varargin{i+1};
                weight = weight(:);
                if length(weight)~=L
                    error('The size of weight is not correct.');
                end
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end
            case 'RHO'
                rho = varargin{i+1};
%             case 'NONNEG'
%                 nonneg = varargin{i+1};
%                 nonneg = nonneg(:);
%                 if ~isscalar(nonneg)
%                     if length(nonneg)~=N
%                         error('Size of nonneg is not right');
%                     end
%                 else
%                     nonneg = ones([N,1])*nonneg;
%                 end
            
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is inconsistent with A or Y');
                end
                if size(x0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(x0,2)~= Ny
                    error('Size of X0 is not valid');
                end
            case 'CONCAVEBASE'
                C = varargin{i+1};
                if any(size(C) ~= [L L])
                    error('CONCAVEBASE is invalid size');
                end
            case 'Z0'
                z0 = varargin{i+1};
                if (size(z0,1) ~= L)
                    error('initial Z is inconsistent with A or Y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                elseif size(z0,2)~= Ny
                    error('Size of Z0 is not valid');
                end
            case 'B0'
                b0 = varargin{i+1};
                if (size(b0,1) ~= L)
                    error('initial Z is inconsistent with A or Y');
                end
                if size(b0,2)==1
                    b0 = repmat(z0,[1,Ny]);
                elseif size(b0,2)~= Ny
                    error('Size of Z0 is not valid');
                end
            case 'D0'
                d0 = varargin{i+1};
                if (size(d0,1) ~= (N+L))
                    error('initial D is inconsistent with A or Y');
                end
                if size(z0,2)==1
                    d0 = repmat(z0,[1,Ny]);
                elseif size(z0,2)~= Ny
                    error('Size of D0 is not valid');
                end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

if b0~=0 && z~=0
    error('B0 and Z0 are both defined');
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C = continuumDictionary(L);
% C = concaveOperator(wv);
% Cinv = C\eye(L);
% s_c = vnorms(Cinv,1);
% Cinv = bsxfun(@rdivide,Cinv,s_c);
% C = bsxfun(@times,C,s_c');
% C = Cinv;
if C==0
    C = continuumDictionary(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 0.01;
ynorms = vnorms(y,1);
tau=ynorms;
if Aisempty
    T = [C];
else
    T = [A C];
end
[U,Sigma1,V] = svd(T);
Sigma = zeros([N+L,1]);
Sigma(1:L) = diag(Sigma1).^2;
Sigmarhoinv = 1./(Sigma + 1);
Q = bsxfun(@times,V,Sigmarhoinv') * V.';
K = Q*T';
Q11 = Q(1:N,1:N);
Q12 = Q(1:N,N+1:N+L);
Q22 = Q(N+1:N+L,N+1:N+L);
K1 = K(1:N,:);
K2 = K(N+1:N+L,:);
kappau = lambda_a.*ones([N,1]);
kappaurho = kappau*(1/rho);
kappav = zeros([L,1]); kappav(1) = -inf; kappav(L) = -inf;
kappar = ones([L,1])./tau;
kapparrho = kappar.*(1/rho);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if b0~=0
    z0 = C\b0;
end

x = zeros([N,Ny]);
z = zeros([L,Ny]);
r = zeros([L,Ny]);
u = x;
v = z;
du = zeros(size(u));
dv = zeros(size(v));
dr = zeros(size(r));

% udu = u-du; vdv = v-dv; rdry = r-dr+y;
% xtmp = Q11*udu + Q12*vdv+K1*rdry;
% ztmp = Q22*vdv + Q12'*udu+K2*rdry;
% % over relaxation
% % update u,v,r
% utmp = max(xtmp+du-kappaurho,0);
% vtmp = max(ztmp+dv,kappav);
% rtmp = soft_thresh(A*xtmp+C*ztmp-y+dr,kapparrho);
% % update the dual variables
% du = du+xtmp-utmp;
% dv = dv+ztmp-vtmp;
% dr = dr+A*xtmp+C*ztmp-rtmp-y;
%     



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
tol_p = sqrt((L*2+N)*Ny)*tol;
tol_d = sqrt((L*2+N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
change_rho = 0;
% idx = [find(nonneg>0)', N+2:N+L-1];
update_rho_active = 1;


while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save t to be used later
    u0 = u; v0 = v; r0 = r;

    % update x,z
    udu = u-du; vdv = v-dv; rdry = r-dr+y;
    x = Q11*udu + Q12*vdv+K1*rdry;
    z = Q22*vdv + Q12'*udu+K2*rdry;
    % over relaxation
    % update u,v,r
    u = max(x+du-kappaurho,0);
    v = max(z+dv,kappav);
    r = soft_thresh(A*x+C*z-y+dr,kapparrho);
    % update the dual variables
    du = du+x-u;
    dv = dv+z-v;
    dr = dr+A*x+C*z-r-y;
    
    if mod(k,2) == 10 || k==1
        % primal feasibility
        res_p = norm(x-u,'fro') + norm(z-v,'fro') + norm(A*x+C*z-r-y,'fro');
        % dual feasibility
        res_d = rho*(norm((u-u0),'fro')+norm(v-v0,'fro')+norm(r-r0,'fro'));
        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0
        % primal feasibility
        res_p = norm(x-u,'fro') + norm(z-v,'fro') + norm(A*x+C*z-r-y,'fro');
        % dual feasibility
        res_d = rho*(norm((u-u0),'fro')+norm(v-v0,'fro')+norm(r-r0,'fro'));
        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
        if update_rho_active
            % update rho
            if res_p > 10*res_d
                rho = rho*3;
                du = du/3; dv = dv/3; dr = dr/3;
                change_rho = 1;
            elseif res_d > 10*res_p
                rho = rho/3;
                du = du*3; dv = dv*3; dr = dr*3;
                change_rho = 1;
            end
            if  change_rho
                kappaurho = kappau.*(1/rho);
                kapparrho = kappar.*(1/rho);
                change_rho = 0;
                fprintf('k=%d,rho=%d\n',k,rho);
            end
        end
    end
    k=k+1;    
end

if Aisempty
    x = [];
else
    x = u;
end
z = v;
du=rho*du;
end
