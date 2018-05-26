function [x,z,C,rho,res_p,res_d] = huwacb_dal_admm(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacb_dal_admm(A,y,wv,varargin)
% L1-constrained hyperspectral unmixing with adaptive concave background (HUWACB) via 
% dual augmented Lagrangian (DAL) with ADMM using residual balancing
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
%     'RHO'     : spectral penalty parameter (default) 0.01
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z
%     d: estimated dual variables (N+L x Ny)
%     rho: spectral penalty parameter at the convergence 
%     res_p,res_d: primal and dual residuals for feasibility

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
%   This is solved via DAL.
%
%        minimize    (1/2) ||y-Ts||^2_F + ||c1.*t||_1 + I_{t>c2}(t)
%           s
%        subject to  s-t = 0
% 
%   Dual problem of the one above is defined as 
%        minimize    f*(aa) + g*(vv)
%          aa,vv
%        subject to  vv = Tt * aa
%  
%   where f*(aa) is the conjugate function of (1/2) ||y-Ts||^2_F and g*(vv)
%   is the conjugate function of ||c1.*t||_1 + I_{t>c2}(t). This problem is
%   solved via ADMM. Its augmented Lagrangian is defined below
%   L = f*(aa) + g*(vv) + d.T(vv-Tt*aa) + 0.5*rho*||(vv-Tt*a)||_2^2
%
%   Refer 
% 
%   Tomioka, R., Sugiyama, M., 2009. Dual-Augmented Lagrangian method for 
%   efficient sparse reconstruction. IEEE Signal Process. Lett. 16, 
%   pp. 1067-1070. https://doi.org/10.1109/LSP.2009.2030111
%
%   for dual formualtion using Fenchel duality theorem, 
%   but alternating minimization and residual balancing is used.
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
% spectral penalty parameter
rho = 0.01;

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
            case 'RHO'
                rho = varargin{i+1};
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
[U,Sigma1,V] = svd(T,'econ');
Sigma = diag(Sigma1).^2;
Sigmarhoinv = 1./(1+rho*Sigma);
Q = bsxfun(@times,U,Sigmarhoinv') * U.';

lambda_a_v = lambda_a.*ones(N,1);
lambda_a_v_rho = lambda_a_v ./ rho;
kappa = zeros(NL,1);
kappa(1:N) = lambda_a_v_rho;
kappa2 = -inf(NL,1);
kappa2(N+1) = 0; kappa2(NL) = 0;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aa = Q*y;
vv = max(min(T'*aa,kappa),kappa2);
d  = T'*aa-vv;



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%a
% tic
tol_p = sqrt(NL*Ny)*tol;
tol_d = sqrt(NL*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
change_rho = 0;
% idx = [find(nonneg>0)', N+2:N+L-1];
update_rho_active = 1;


while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 0 || k==1
        vv0 = vv;
    end
      
%     fprintf('k=%d\n',k);
    % update aa
    aa = Q * (y + rho*T*(vv-d));
    % update vv
    vv = max(min(T'*aa+d,kappa),kappa2);
    
    % update the dual variables
    d = d + T'*aa-vv;
    
%     if mod(k,10) == 0 || k==1
%         % primal feasibility
%         res_p = norm(s-t,'fro');
%         % dual feasibility
%         res_d = rho*(norm((t-t0),'fro'));
%         if  verbose
%             fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
%         end
%     end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0
        % primal feasibility
        res_p = norm(T'*aa-vv,'fro');
        % dual feasibility
        res_d = rho*(norm(T*(vv-vv0),'fro'));
        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
        % primal feasibility
        res_p = norm(T'*aa-vv,'fro');
        % dual feasibility
        res_d = rho*(norm(T*(vv-vv0),'fro'));
        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
        if update_rho_active
            % update rho
            if res_p > 10*res_d
                rho = rho*2;
                d = d/2;
                change_rho = 1;
            elseif res_d > 10*res_p
                rho = rho/2;
                d = d*2;
                change_rho = 1;
            end
            if  change_rho
                Sigmarhoinv = 1./(1+rho*Sigma);
                Q = bsxfun(@times,U,Sigmarhoinv') * U.';
                kappa(1:N) = lambda_a_v ./ rho;
                change_rho = 0;
%                 fprintf('k=%d,rho=%d\n',k,rho);
            end
        end
    end
    k=k+1;    
end

d = rho*d;

x = d(1:N,:);
z = d(N+1:NL,:);

end