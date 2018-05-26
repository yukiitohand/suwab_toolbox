function [x,z,C,sol,outs] = huwacb_aradmm2(A,y,wv,varargin)
% [x,z,C,sol,outs] = huwacb_aradmm2(A,y,wv,varargin)
% L1-constrained hyperspectral unmixing with adaptive concave background (HUWACB) via 
% an adaptive relax alternating direction method of multipliers (ARADMM)
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
%     'VERBOSE' : 0: no print
%                 1: print every iteration
%                 2: evaluate objective every iteration
%                 3: more print out for debugging adaptive relaxed ADMM
%                 (default) 0
%     'rho'     : initial spectral penalty parameter (tau in the paper), scalar
%                 (default) 0.01
%     'ADP_FLAG': Option for determination method of the spectral 
%                 penalty parameter {0,1,3,5}
%                 0: no adaptation, fixed rho(tau)
%                 1: Adaptive ADMM (AADMM)
%                 3: residual balancing
%                 5: Adaptive Relaxed ADMM (ARADMM)
%                 (default) 5
%     'GAMMA'   : Relaxation parameter, scalar
%                 Over relaxation if GAMMA > 1, typically 1.5
%                 (default) 1
%                 *With ARADMM, this is just an initial GAMMA and adjusted
%                 automatically after that.
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
%     'D0'      : Initial dual parameters [N+L,Ny] (non-scaling form)
%                 (default) 0
%
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z
%     sol: solution, "see aradmm_core.m"
%     outs: see "aradmm_core.m"
%
%  L1-constrained HUWACB solves the following convex optimization problem 
%  
%         minimize    (1/2) ||y-Ax-Cz||^2_F + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
%  The base of ADMM formulation is same as "huwacb_admm2.m"
%
% References
%  Xu, Z., Figueiredo, M.A.T., Yuan, X., Studer, C., Goldstein, T., 2017. 
%   Adaptive relaxed ADMM: Convergence theory and practical implementation, 
%   in: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 
%   pp. 7234?7243. https://doi.org/10.1109/CVPR.2017.765
%  Xu, Z., Figueiredo, M.A.T., Goldstein, T., 2017. Adaptive ADMM with 
%   spectral penalty parameter selection, in: Proceedings of the 20th 
%   International Conference on Artificial Intelligence and Statistics 
%   (AISTATS), PMLR. pp. 718?727.
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
NL = N+L;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = 2000;
% display only sunsal warnings
verbose = 0;
% tolerance for the primal and dual residues
tol = 1e-4;
% sparsity constraint on the library
lambda_a = 0.0;
% rho (tau in the paper)
rho = 0.01;
% adp_flg
adp_flag = 5;
% gamma
gamma = 1;

% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
% initialization of B
b0 = 0;
% initial dual parameters
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
                verbose = varargin{i+1}; 
            case 'RHO'
                rho = varargin{i+1};
            case 'ADP_FLAG'
                adp_flag = varargin{i+1};
            case 'GAMMA'
                gamma = varargin{i+1};
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end            
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
                if (size(d0,1) ~= (NL))
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

% option for the input to aradmm_core.m
opts = [];
opts.tol = tol; %stop criterion, relative tolerance
opts.maxiter = maxiter; %max interation
opts.tau = rho; %initial stepsize
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration
%3: more print out for debugging adaptive relaxed ADMM
opts.verbose = verbose; 
opts = get_default_opts(opts);
opts.adp_flag = adp_flag;
opts.gamma = gamma;

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
if Aisempty
    T = [C];
else
    T = [A C];
end

[~,Sigma1,V] = svd(T);
Sigma = zeros([NL,1]);
Sigma(1:L) = diag(Sigma1).^2;
Q = @(tau) bsxfun(@times,V,1./(Sigma + tau)') * V.';

ayy = T' * y;

lambda_a_v = lambda_a.*ones([N,1]);
kappa = zeros([NL,1]);
kappa(1:N) = lambda_a_v;
kappa2 = zeros([NL,1]);
kappa2(N+1) = -inf; kappa2(NL) = -inf;

%%
h = @(u) 0.5*norm(y-T*u,'fro').^2;
g = @(v) sum(kappa.*v);

%objective
obj = @(u, v) h(u)+g(v);
% 
solvh = @(v, l, tau1) Q(tau1)*(ayy+tau1*v+l); %update u
solvg = @(au, l, tau1) max(au-l/tau1-kappa./tau1,kappa2);  %update v
fA = @(x) x;
fAt = @(x) x;
fB = @(x) -x;
fb = 0;

opts.obj = obj;

%% initialization
s0 = zeros(NL,1);
if x0~=0
    s0(1:N,:) = x0;
end
if b0~=0 && C~=0
    s0(N+1:NL,:) = C\b0;
end
if z0~=0
    s0(N+1:NL,:) = z0;
end
l0 = zeros(size(x0));
if d0~=0
    l0 = d0;
end

%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, s0, l0, opts);
outs.runtime  = toc;

x = sol.v(1:N,:);
z = sol.v(N+1:NL,:);

end
