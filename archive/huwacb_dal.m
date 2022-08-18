function [x,z,C] = huwacb_dal(A,y,wv,varargin)
% [x,z,C] = huwacb_dal(A,y,wv,varargin)
% L1-constrained hyperspectral unmixing with adaptive concave background (HUWACB) via 
% dual augmented Lagrangian (DAL) 
%
%  Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x 1) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
%  Optional parameters
%     'TOL': tolearance (default) 1e-5
%     'MAXITER' : maximum number of iterations (default) 100
%     'VERBOSE' : 0: none,                  1: only the last, 
%                 2: every outer iteration, 3: every inner iteration
%                 (default) 0
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%     'SOLVER'  : internal solver for DAL. Can be either:
%                 'nt'   : Newton method with cholesky factorization
%                 'ntsv' : Newton method memory saving (slightly slower)
%                 'cg'   : Newton method with PCG (default)
%                 'qn'   : Quasi-Newton method
%  Outputs
%     x: estimated abundances (N x 1)
%     z: estimated concave background (L x 1)
%     C: matrix (L x L) for z
%
%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz||^2_F + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
%
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
%    This is solved via DAL.
%
%  References
%  Tomioka, R., Sugiyama, M., 2009. Dual-Augmented Lagrangian method for 
%   efficient sparse reconstruction. IEEE Signal Process. Lett. 16, 
%   pp. 1067-1070. https://doi.org/10.1109/LSP.2009.2030111
%  Tomioka, R., Suzuki, T., Sugiyama, M., 2011. Super-Linear convergence 
%   of dual augmented-Lagrangian algorithm for sparsity regularized 
%   estimation. J. Mach. Learn. Res. 12, pp. 1537?1586.
%  Tomioka, R., Suzuki, T., Sugiyama, M., 2011. Augmented Lagrangian 
%   methods for learning, selecting, and combining, in: Sra, S., Nownzin, 
%   S., Wright, S.J. (Eds.), Optimization for Machine Learning. MIT Press, 
%   pp. 255?285.
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
if Ny~=1
    error('This function only works with y with one column');
end
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
% sparsity constraint on the library
lambda_a = 0.0;
% maximum number of AL iteration
maxiter = 100;
% display only sunsal warnings
verbose = 0;
solver = 'cg';
% tolerance for the primal and dual residues
tol = 1e-5;

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
            case 'SOLVER'
                solver = varargin{i+1};
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

if b0~=0 
    if z0~=0
        error('B0 and Z0 are both defined');
    end
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

lambda_a_v = lambda_a.*ones([N,1]);
kappa = zeros([NL,1]);
kappa(1:N) = lambda_a_v;
kappa2 = true([NL,1]);
kappa2(N+1) = false; kappa2(NL) = false;

ww0 = zeros(NL,1);

[ww,status] = dalsql1pn(ww0,T,y,1,kappa,kappa2,'display',verbose,...
                'solver',solver,'stopcond','pdg','tol',tol,...
                'maxiter',maxiter);

x = ww(1:N,:);
z = ww(N+1:NL,:);

end