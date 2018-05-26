function [x,z,C] = huwacb_fista_backtracking(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacb_fista_backtracking(A,y,wv,varargin)
% L1-constrained hyperspectral unmixing with adaptive concave background (HUWACB) via 
% fast iterative shrinkage thresholding algorithm (FISTA) with backtracking
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
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz||^2_F + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
%  The problem will be converted to a variant of non-negative lasso:
%
%         minimize    (1/2) ||y-Ts||^2_F + ||c1.*s||_1
%           s
%         subject to  s >= c2
%   where       _   _ 
%          s = |  x  |,  T = [A C],
%              |_ z _|                       _       _
%                 _          _              |    0_L  |
%          c_1 = |  lambda_a  |,  and  c2 = |   -inf  | 
%                |_   0_L    _|             | 0_{L-2} |
%                                           |_  -inf _|
%
%   This problem is solved via FISTA with backtracking.
%   Reference
%     Beck, A., Teboulle, M., 2009. A fast iterative shrinkage-thresholding
%      algorithm. Soc. Ind. Appl. Math. J. Imaging Sci. 2, 183?202. 
%      https://doi.org/10.1137/080716542
%
%  For coding, https://github.com/tiepvupsu/FISTA
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
maxiter = 1000;
% display only sunsal warnings
verbose = false;
% tolerance for the primal and dual residues
tol = 1e-4;
% sparsity constraint on the library
lambda_a = 0.0;
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
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end
            case 'CONCAVEBASE'
                C = varargin{i+1};
                if any(size(C) ~= [L L])
                    error('CONCAVEBASE is invalid size');
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
if Aisempty
    T = [C];
else
    T = [A C];
end
L = 1;
% [~,Sigma1,~] = svd(T);
% Sigma = zeros([NL,1]);
% Sigma(1:L) = diag(Sigma1).^2;
% L = Sigma(1); Linv = 1/L;
ayy = T' * y;

lambda_a_v = lambda_a.*ones(N,1);
kappa_ = zeros(NL,1);
kappa_(1:N) = lambda_a_v;
kappa2 = zeros(NL,1);
kappa2(N+1) = -inf; kappa2(NL) = -inf;

% function place holders
TtT = T.'*T;
grad = @(x1) TtT*x1 - ayy;
proj = @(x1,L1) max(x1-kappa_/L1,kappa2);
F = @(x1) 0.5*norm(T*x1-y,'fro')^2 + sum(sum((kappa_.*x1)));
Q = @(x1,y1,L1) 0.5*norm(T*y1-y,'fro')^2 + sumel(grad(y1).*(x1-y1)) + 0.5*L1*norm(x1-y1,'fro')^2 + sum(sum((kappa_.*x1)));

%% initialization
y_old = zeros(NL,Ny); x_old = y_old;
t_old = 1;
k = 1;
eta = 2;
%%
cost_old = F(y_old);
while  k <= maxiter
    Lbar = L; 
    while true 
        zk = proj(y_old-1/Lbar*grad(y_old),Lbar);
        valF = F(zk);
        valQ = Q(zk, y_old, Lbar);
        if valF <= valQ 
            break;
        end
        Lbar = Lbar*eta; 
        L = Lbar; 
    end 
    x_new = proj(y_old-1/L*grad(y_old),L);
    t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
    y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
    %% check stop criteria
    e = norm1(x_new - x_old)/numel(x_new);
    if e < tol
        break;
    end
    if verbose && (k==1 || mod(k,10)==0)
        cost_new = F(x_new);
        fprintf('k=%3d, L=%f, cost_new=%f, cost_old=%f\n',k,L,cost_new,cost_old);
        cost_old = cost_new;
    end
    %% update
    x_old = x_new;
    t_old = t_new;
    y_old = y_new;
    
    k = k+1;

end
x = x_new(1:N,:);
z = x_new(N+1:NL,:);
end