function [x,z,C,r,d,rho,res_p,res_d] = huwacbl1_gadmm3(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacbl1_admm3_indrho(A,y,wv,varargin)
% L1-constraind hyperspectral unmixing with adaptive concave background (HUWACB)
% based on Least absolute deviation.
% via a generalized alternating direction method of multipliers (ADMM).
% 
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
Rhov = ones(NL+L,1);
if Aisempty
    T = [C];
else
    T = [A C];
end
K = [eye(NL);T];
% KtPKinv = (K'*diag(Rhov)*K) \ eye(NL);
% KtPKinvKtP = KtPKinv*K'*diag(Rhov);
ybar = [zeros(NL,Ny);y];
% KtPy = K'*diag(Rhov)*ybar;
% KtPKinvKtPy = KtPKinv * KtPy;
KtP = K'.*Rhov';
KtPKinv = (KtP*K) \ eye(NL);
KtPKinvKtP = KtPKinv*KtP;
KtPy = KtP*ybar;
KtPKinvKtPy = KtPKinv * KtPy;
% projection operator
c1 = zeros([N+2*L,Ny]);
c1(1:N,:) = lambda_a.*ones([N,Ny]);
c1(N+L+1:N+L*2,:) = ones([L,1])./tau;
% c1(N+L+1:N+L*2) = ones([L,1])*tau1;
c1rho = c1./rho./Rhov;

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
                s = KtPKinvKtPy;
                Ks_minus_ybar = K*s-ybar;
                t = max(soft_thresh(Ks_minus_ybar,c1rho),c2);
%                 s = t;
                d = Ks_minus_ybar-t;
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
tol_p = sqrt((L*2+N)*Ny)*tol;
tol_d = sqrt((L*2+N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;

% idx = [find(nonneg>0)', N+2:N+L-1];
update_rho_active = 1;
onesNy1 = ones(Ny,1);
ones1NL2 = ones(1,NL+L);

Ks_minus_ybar = zeros(NL+L,Ny);
idx_Ks_s = false(NL+L,1); idx_Ks_s(1:NL) = true;
idx_Ks_r = false(NL+L,1); idx_Ks_r(NL+1:NL+L) = true;

Ks = zeros(NL+L,Ny);
Ksq = sum(K.^2,2);
while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save t to be used later
    if mod(k,10) == 0 ||k==1
        t0 = t; s0 = s;
    end

    % update s
    s = KtPKinvKtP * (t-d) + KtPKinvKtPy;
%     tic; Ks_minus_ybar = K*s-ybar; toc;
    Ks_minus_ybar = calc_Ksmy(s,Ks_minus_ybar,y,idx_Ks_s,idx_Ks_r,T);
    % update t
    t = max(soft_thresh(Ks_minus_ybar+d,c1rho),c2);
    % update the dual variables
    d = d + Ks_minus_ybar-t;

    if mod(k,10) == 0 || k==1
        tt0 = (t-t0); res_dmat = KtP*tt0.*rho;
%         Ksqtt0 = Ksq.*(t-t0).^2;
        % primal feasibility
        res_p = norm(Ks_minus_ybar-t,'fro');        
        % dual feasibility
        res_d = norm(res_dmat,'fro');
%         res_d = sqrt(Rhov'.^2*Ksqtt0*rho'.^2);
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0
        st2 = (Ks_minus_ybar-t).^2;
        % primal feasibility
        res_pv = sqrt(ones1NL2*st2);
        % dual feasibility
%         res_dv = rho.*sqrt(Rhov'.^2*tt02);
        res_dv = sqrt(sum(res_dmat.^2,1));
%         res_dv = rho.*sqrt(Rhov'.^2*Ksqtt0);
%         if k==390
%             k;
%         end
        
        % update rho
        idx = res_pv > 10*res_dv;
        if any(idx)
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
        end
        idx2 = res_dv > 10*res_pv;
        if any(idx2)
            rho(idx2) = rho(idx2)/2;
            d(:,idx2) = d(:,idx2)*2;
        end
        c1rho = c1./rho; 
        % Rho for different dimension
        
        if (k<300)
            % Rho for different dimension
            % primal feasibility
            res_pv2 = sqrt(st2*onesNy1);
            % dual feasibility
            res_dv2 = sqrt(sum(((Rhov.*(calc_Ks(s-s0,Ks,idx_Ks_s,idx_Ks_r,T)).*rho)).^2,2));
%             res_dv2 = Rhov .* sqrt(Ksqtt0*rho'.^2);
            idx3 = res_pv2 > 10*res_dv2;
            Rhov(idx3) = Rhov(idx3)*2;
            d(idx3,:) = d(idx3,:)/2;
            idx4 = res_dv2 > 10*res_pv2;
            Rhov(idx4) = Rhov(idx4)/2;
            d(idx4,:) = d(idx4,:)*2;
            if any(idx3) || any(idx4)
                KtP = K'.*Rhov';
                KtPKinv = (KtP*K) \ eye(NL);
                KtPKinvKtP = KtPKinv*KtP;
                KtPy = KtP*ybar;
                KtPKinvKtPy = KtPKinv * KtPy;
            end
                      
        end
        c1rho = c1rho./Rhov;
    end
    k=k+1;    
end

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
r = t(N+L+1:N+L*2,:)*tau1;
d=rho.*d;
end

function [Ks_minus_ybar] = calc_Ksmy(s,Ks_minus_ybar,y,idx_Ks_s,idx_Ks_r,T)
Ks_minus_ybar(idx_Ks_s,:) = s;
Ks_minus_ybar(idx_Ks_r,:) = T*s-y;
end

function [Ks] = calc_Ks(s,Ks,idx_Ks_s,idx_Ks_r,T)
Ks(idx_Ks_s,:) = s;
Ks(idx_Ks_r,:) = T*s;
end

