function [x,z,C,r,d,rho,res_p,res_d] = huwacbl1_admm1_1_indconverge(A,y,wv,varargin)
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
rho = 0.01*ones([1,Ny]);
% rho = ones([N+L*2,Ny]);

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
    C = C*2;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 0.01;
ynorms = vnorms(y,1);
tau=ynorms;
tau1 = 0.2;
if Aisempty
    T = [C tau1*eye(L)];
else
    T = [A C tau1*eye(L)];
end
TTt = T*T';
Tpinvy = T' * (TTt \ y);
PT_ort = eye(N+2*L) - T' * (TTt \ T);
% projection operator
c1 = zeros([N+2*L,Ny]);
c1(1:N,:) = lambda_a.*ones([N,Ny]);
c1(N+L+1:N+L*2,:) = ones([L,1])./tau*tau1;
% c1(N+L+1:N+L*2) = ones([L,1])*tau1;
c1rho = c1./rho;

c2 = zeros([N+2*L,1]);
c2(N+1) = -inf; c2(N+L) = -inf; c2(N+L+1:N+2*L) = -inf;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if b0~=0
    z0 = C\b0;
end
if ~Aisempty
    if x0==0
        if z0==0
            if d0==0
                s = Tpinvy;
                t = max(soft_thresh(s,c1rho),c2);
%                 s = t;
                d = s-t;
            end
        end
    else
        d=d0/rho;
        t = [x0;z0;y-A*x0-C*z0];
        s = max(soft_thresh(t-d,c1rho),c2);
        d = d+s-t;
%     else
%         error('Manual initial variable setting is all or nothing');
    end
else
    error('not implemented yet');
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
tol_p = sqrt((L*2+N))*tol;
tol_d = sqrt((L*2+N))*tol;
k=1;
res_p = inf;
res_d = inf;
% idx = [find(nonneg>0)', N+2:N+L-1];
update_rho_active = 1;
actv_idx = true(1,Ny);

while (k <= maxiter) && any(actv_idx)
    % save t to be used later
    if mod(k,10) == 0
        t0 = t;
    end

    % update t
    s(:,actv_idx) = PT_ort * (t(:,actv_idx)-d(:,actv_idx)) + Tpinvy(:,actv_idx);
%     s = 0.5*s+0.5*t; % over relaxation
    % update s
    t(:,actv_idx)= max(soft_thresh(s(:,actv_idx)+d(:,actv_idx),c1rho(:,actv_idx)),c2);
    % update the dual variables
    d(:,actv_idx) = d(:,actv_idx) + s(:,actv_idx)-t(:,actv_idx);
%     if mod(k,2) == 10 || k==1
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
        res_pv = vnorms(s-t,1);
        % dual feasibility
        res_dv = rho .* vnorms(t-t0,1);
        
        if update_rho_active
            % update rho
            idx = res_pv > 10*res_dv;
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
            idx2 = res_dv > 10*res_pv;
            rho(idx2) = rho(idx2)/2;
            d(:,idx2) = d(:,idx2)*2;
            c1rho = c1./rho;
        end
        actv_idx = or(abs(res_pv)>tol_p, abs(res_dv)>tol_d);
        
        res_p = norm(res_pv); res_d = norm(res_dv);

        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
    end
    k=k+1;    
end

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
r = t(N+L+1:N+L*2,:);
d=rho.*d;
end