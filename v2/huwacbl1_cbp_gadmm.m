function [x,z,C,r,d,rho,Rhov,res_p,res_d] = huwacbl1_cbp_gadmm(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacbl1_gadmm_a(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% a generalized alternating direction method of multipliers (ADMM)
%
%  HUWACB solves the following convex optimization  problem 
%  
%      minimize  ||lambda_r.*(y-Ax-Cz)||_1 + ||lambda_a.*x||_1
%        x,z                                        + ||lambda_c.*z||_1
%      subject to  x>=0 and z>=c2_z
%  where by default,C(wv) is the collection of bases to represent the 
%  concave background:
%                _     _
%               | -inf  |  /\
%               |   0   |  ||
%       c2_z =  |   :   |  || Nc
%               |   0   |  ||
%               |_-inf _|  \/
%  
%  You can optionally change C and c2_z as you like. By default,
%  lambda_r = 1, lambda_a = 0.01, and lambda_c = 0. So the default problem
%  is
%      minimize  ||y-Ax-Cz||_1 + ||lambda_a.*x||_1
%        x,z                                       
%      subject to  x>=0 and z>=c2_z
%  
%  A variable is augmented r=y-Ax-Cz and the problem is casted as a
%  constrained sparse least absolute deviation, subsequently as a
%  constrained basis pursuit (CBP).
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
%     'LAMBDA_A': sparsity constraint on x, the size needs to be compatible
%                 with the operation (lambda_a.*x)
%                 (default) 0.01
%     'LAMBDA_C': sparsity constraint on z, the size needs to be compatible
%                 with the operation (lambda_c.*z)
%                 (default) 0
%     'LAMBDA_R': weight coefficients for the residual r. the size needs to
%                 be comaptible with the operation (lambda_r .* r)
%                 (default) 1
%     'c2_z'    : right hand side of the c2_z, compatible with the
%                 operation (z >= c2_z). By default 
%     'X0'      : Initial x (coefficient vector/matrix for the libray A)
%                 (default) []
%     'Z0'      : Initial z (coefficient vector/matrix for the concave
%                 bases C) (default) []
%     'C'       : Concave bases C [L x L] or any base matrix [L x Nc]. 
%                 This will be created from 'wv'  if not provided
%     'B0'      : Initial Background vector B [L x N]. This will be converted to C
%                 (default) []
%     'R0'      : Initial 'r' residual
%                 (default) []
%     'D0'      : Initial dual parameters [N+Nc+L,Ny] (non-scaling form)
%                 (default) []
%     'rho'     : initial spectral penalty parameter for different samples,
%                 scalar or the size of [1,Ny]
%                 (default) 0.01
%     'Rhov'    : initial spectral penalty parameter, for different
%                 dimensions. scalar or the size of [L,1]
%                 (default) 1
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (Nc x Ny)
%     C: matrix (L x L) for z
%     r: residual (L x Ny) (not exactly equal to (y-Ax-Cz), due to practical convergence limitation)
%     d: estimated dual variables ( (N+Nc+L) x Ny )
%     rho: spectral penalty parameter "rho" at the convergence, [1 Ny]
%     Rhov: spectral penalty parameter "Rhov" at the convergence, [L, 1]
%     res_p,res_d: primal and dual residuals for feasibility

% note
% error('inputParser is still in progress.');

%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
maxiter_default = 1000;
% display only sunsal warnings
verbose_default = false;
% tolerance for the primal and dual residues
tol_default = 1e-4;
% sparsity constraint on the library
lambda_a_default = 0.01;
% sparsity constraint on the concave base matrix
lambda_c_default = 0.0;
% weight coefficient for the model residual
lambda_r_default = 1.0;
% spectral penalty parameter
rho_default = 0.01*ones([1,Ny]);
Rhov_default = ones(N+L*2,1);

% initialization of X0
x0_default = [];
% initialization of Z0
z0_default = [];
% initialization of B
b0_default = [];
% initialization of r0
r0_default = [];
% initialization of Lagrange multipliers, d0
d0_default = [];
% base matrix of concave curvature
C_default = [];
% c2_z
c2_z_default = [];

p = inputParser;
% p.KeepUnmatched = true;
addOptional(p,'maxiter',maxiter_default);
addOptional(p,'verbose',verbose_default);
addOptional(p,'tol',tol_default);
addOptional(p,'Lambda_A',lambda_a_default);
addOptional(p,'Lambda_C',lambda_c_default);
addOptional(p,'Lambda_R',lambda_r_default);
addOptional(p,'X0',x0_default);
addOptional(p,'Z0',z0_default);
addOptional(p,'B0',b0_default);
addOptional(p,'R0',r0_default);
addOptional(p,'ConcaveBase',C_default);
addOptional(p,'c2_z',c2_z_default);
addOptional(p,'D0',d0_default);
addOptional(p,'rho',rho_default);
addOptional(p,'Rhov',Rhov_default);

% addOptional(p,'Lambda_A',lambda_a_default,@isnumeric);
% addOptional(p,'Lambda_C',lambda_c_default,@isnumeric);
% addOptional(p,'Lambda_R',lambda_r_default,@isnumeric);
% addOptional(p,'X0',x0_default, @(x) validateattributes(x,{'size',[N,Ny]},mfilename,'X0'));
% addOptional(p,'Z0',z0_default,@isnumeric);
% addOptional(p,'B0',b0_default, @(x) validateattributes(x,{'size',[L,Ny]},mfilename,'B0'));
% addOptional(p,'R0',r0_default,@isnumeric);
% addOptional(p,'ConcaveBase',C_default,@(x) validateattributes(x,{'nrow',L},mfilename,'ConcaveBase'));
% addOptional(p,'c2_z',c2_z_default,@isnumeric);
% addOptional(p,'D0',d0_default, @isnumeric);
% addOptional(p,'rho',rho_default,@isnumeric);
% addOptional(p,'Rhov',Rhov_default,@isnumeric);
% addOptional(p,'maxiter',maxiter_default,@isscalar);
% addOptional(p,'verbose',verbose_default,@(x) or(islogical(x),or(x==0,x==1)));
% addOptional(p,'tol',tol_default,@isscalar);

parse(p,varargin{:});
lambda_a = p.Results.Lambda_A;
lambda_c = p.Results.Lambda_C;
lambda_r = p.Results.Lambda_R;
x0 = p.Results.X0;
z0 = p.Results.Z0;
b0 = p.Results.B0;
r0 = p.Results.R0;
d0 = p.Results.D0;
C = p.Results.ConcaveBase;
c2_z = p.Results.c2_z;
rho = p.Results.rho;
Rhov = p.Results.Rhov;
maxiter = p.Results.maxiter;
verbose = p.Results.verbose;
tol = p.Results.tol;

% if isvector(lambda_a), lambda_a = lambda_a(:); end
% if isvector(lambda_c), lambda_c = lambda_c(:); end
% if isvector(lambda_r), lambda_r = lambda_r(:); end
% if isvector(Rhov), Rhov = Rhov(:); end
% if isvector(rho), rho = rho(:)'; end
% if isvector(c2_z), c2_z = c2_z(:); end

% Create the bases for continuum.
if isempty(C)
    C = continuumDictionary(wv);
    % C = continuumDictionary_smooth(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
    C = C*2;
end
[Lc,Nc] = size(C);
if Lc~=L
    error('size of given base is not correct');
end

if isempty(c2_z)
    c2_z = zeros([Nc,1]);
    c2_z(1) = -inf; c2_z(Nc) = -inf;
end

try
    lambda_a = lambda_a.*ones([N,Ny]);
catch
    error('size of lambda_a does not seem to be right');
end

try
    lambda_c = lambda_c.*ones([Nc,Ny]);
catch
    error('size of lambda_c does not seem to be right');
end

try
    lambda_r = lambda_r.*ones([L,Ny]);
catch
    error('size of lambda_r does not seem to be right');
end

try
    c2_z = c2_z.*ones(Nc,1);
catch
    error('size of c2_z does not seem to be right');
end

if ~isempty(b0) && ~isempty(z0)
    error('B0 and Z0 are both defined');
end


%%
% pre-processing of the matrix
ynorms = vnorms(y,1);
tau=ynorms;
tau1 = 0.2;
if Aisempty
    G = [C tau1*eye(L)];
else
    G = [A C tau1*eye(L)];
end
% projection operator
c1 = zeros([N+Nc+L,Ny]);
c1(1:N,:) = lambda_a;
c1(N+1:N+Nc,:) = lambda_c;
c1(N+Nc+1:N+Nc+L,:) = lambda_r./tau*tau1;

c2 = zeros([N+Nc+L,1]);
c2(N+1:N+Nc) = c2_z;
c2(N+Nc+1:N+Nc+L) = -inf;

s0 = [x0;z0;r0];

%%
% perform main-loop
[s,t,d,rho,Rhov,res_p,res_d] = cbp_gadmm(G,y,c1,c2,'X0',s0,'D0',d0,...
    'rho',rho,'Rhov',Rhov,'maxiter',maxiter,'tol',tol,'verbose',verbose);

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+Nc,:);
r = t(N+Nc+1:N+Nc*2,:)*tau1;

end
