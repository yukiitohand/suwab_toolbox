function [x,z,r,d,rho,Rhov,res_p,res_d,cost_val]...
    = huwacbl1_admm_gat_a_batch(A,y,C,varargin)
% [x,z,res_p,res_d] = uwacbl1_admm_gat_a_batch(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% a generalized alternating direction method of multipliers (ADMM)
%
%  Inputs
%     A : Each of the page is a dictionary matrix (L x Na x S) where Na is
%         the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x Ny) where N is the number of the
%     observations.
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
%     'R0'      : Initial 'r'
%                 (default) 0
%     'D0'      : Initial dual parameters [N+L,L] (non-scaling form)
%                 (default) 0
%     'rho'     : initial spectral penalty parameter for different samples,
%                 scalar or the size of [1,Ny]
%                 (default) 0.01
%     'Rhov'    : initial spectral penalty parameter, for different
%                 dimensions. scalar or the size of [L,1]
%                 (default) 1
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z
%     r: residual
%     d: estimated dual variables (N+L x Ny)
%     rho: spectral penalty parameter "rho" at the convergence, [1 Ny]
%     Rhov: spectral penalty parameter "Rhov" at the convergence, [L, 1]
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
% if (nargin-length(varargin)) ~= 3
%     error('Wrong number of required parameters');
% end
% % mixing matrixsize
% Aisempty = isempty(A);
% if Aisempty
%     N = 0;
% else
%     [LA,N] = size(A);
% end
% % data set size
% [L,Ny] = size(y);
% if ~Aisempty
%     if (LA ~= L)
%         error('mixing matrix M and data set y are inconsistent');
%     end
% end

[L,Ny,M] = size(y);
[~,N,~] = size(A);
[~,Nc,~] = size(C);

%%
%tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = int32(1000);
% display only sunsal warnings
verbose = false;
% tolerance for the primal and dual residues
tol = single(1e-4);
% sparsity constraint on the library
lambda_a = 0.0;
lambda_c = 0;
lambda_r = 1.0;
% spectral penalty parameter
rho = 0.01*ones(1,Ny,M,'gpuArray');
Rhov = ones(N+Nc+L,1,M,'gpuArray');

% initialization of X0
x0 = [];
% initialization of Z0
z0 = [];
% initialization of r0
r0 = [];
% initialization of Lagrange multipliers, d0
d0 = [];

precision = 'single';
isdebug = false;

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
            case 'LAMBDA_C'
                lambda_c = varargin{i+1};
            case 'LAMBDA_R'
                lambda_r = varargin{i+1};
            case 'RHO'
                rho = varargin{i+1};
            case 'RHOV'
                Rhov= varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
            case 'C2_Z'
                c2_z = varargin{i+1};
            case 'Z0'
                z0 = varargin{i+1};
            case 'R0'
                r0 = varargin{i+1};
            case 'D0'
                d0 = varargin{i+1};
            case 'PRECISION'
                precision = varargin{i+1};
            case 'DEBUG'
                isdebug   = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

if strcmpi(precision,'single')
    rho = single(rho); Rhov = single(Rhov); lambda_r = single(lambda_r);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(C)
    error('Specify C');
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 0.01;
ynorms = vnorms(y,1);
tau=single(ynorms);

tau1 = 0.2;
T = cat(2,A,C,tau1*repmat(eye(L,precision,'gpuArray'),[1,1,M]));

I_NNcL = eye(N+Nc+L,precision,'gpuArray');
PinvTt = pagefun(@transpose,T)./Rhov;

TPinvTt = pagefun(@mtimes,T,PinvTt);
PinvTt_invTPinvTt = pagefun(@mrdivide,PinvTt,TPinvTt);
Tpinvy =  pagefun(@mtimes,PinvTt_invTPinvTt,y);
PT_ort = I_NNcL - pagefun(@mtimes,PinvTt_invTPinvTt,T);

% TPinvTt = zeros(L,L,M,precision,'gpuArray');
% for mi=1:M, TPinvTt(:,:,mi) = T(:,:,mi)*PinvTt(:,:,mi); end
% PinvTt_invTPinvTt = pagefun(@mrdivide,PinvTt,TPinvTt);
% Tpinvy = zeros(N+Nc+L,Ny,M,precision,'gpuArray');
% for mi=1:M, Tpinvy(:,:,mi) = PinvTt_invTPinvTt(:,:,mi) * y(:,:,mi); end
% PT_ort = zeros(N+Nc+L,N+Nc+L,M,precision,'gpuArray');
% for mi=1:M, PT_ort(:,:,mi) = I_NNcL - PinvTt_invTPinvTt(:,:,mi)*T(:,:,mi); end

% projection operator
c1 = zeros([N+Nc+L,Ny,M],precision,'gpuArray');
c1(1:N,:,:) = lambda_a.*ones(N,Ny,M,precision,'gpuArray');
c1(N+1:N+Nc,:,:) = lambda_c.*ones(Nc,Ny,M,precision,'gpuArray');
c1(N+Nc+1:N+Nc+L,:,:) = lambda_r.*ones(L,Ny,M,precision,'gpuArray')./tau*tau1;
c1rho = c1./rho./Rhov;


c2 = zeros([N+Nc+L,1],precision,'gpuArray');
c2(N+1:N+Nc,1) = c2_z;
c2(N+Nc+1:N+Nc+L,1) = -inf;

clear lambda_a lambda_c lambda_r A C ynorms 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(x0) && isempty(z0) && isempty(r0) && isempty(d0)
    s = Tpinvy;
    t = max(soft_thresh(s,c1rho),c2);
    d = s-t;
elseif ~isempty(x0) && ~isempty(z0) && ~isempty(r0) && ~isempty(d0)
    d=d0./rho./Rhov;
    t = cat(1,x0,z0,r0);
    % t = max(soft_thresh(s+d,c1rho),c2);
    % d = d+s-t;
else 
    error('not implemented yet. Initialization works with all or nothing.');
end

clear x0 z0 r0
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
tol_p = single(sqrt((N+Nc+L)*Ny)*tol);
tol_d = single(sqrt((N+Nc+L)*Ny)*tol);
k=1;
res_p = single(inf);
res_d = single(inf);
onesNy1 = ones(Ny,1,precision,'gpuArray');
ones1NNcL = ones(1,N+Nc+L,precision,'gpuArray');
ones1NyM = ones(1,Ny,M,precision,'gpuArray');
onesNNcL1M = ones(N+Nc+L,1,M,precision,'gpuArray');

Tcond = ones(1,1,M,precision,'gpuArray');
for i=1:M
    Tcond(i) = cond(T(:,:,i)*T(:,:,i)');
end
thRconv_s = 1e-10./Tcond;
thRconv_b = 1e+10./Tcond;
switch lower(precision)
    case 'double'
        th_cond = 1e13;
    case 'single'
        th_cond = 1e6;
end

if isdebug
    cost_vals = [];
    params = [];
    params_2 = [];
    Cnd_Val = ones(1,1,M,precision,'gpuArray');
    for i=1:M
        Cnd_Val(i) = cond(TPinvTt(:,:,i)*TPinvTt(:,:,i)');
    end
    Cnd_Val_apro = Tcond.*max(Rhov,[],1)./min(Rhov,[],1);
end

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d))
    if isdebug
        cost_val = sum(abs(y-pagefun(@mtimes,T(:,1:N+L),t(1:N+L,:)))./tau.*lambda_r,'all')...
            +sum(abs(lambda_a.*t(1:N,:)),'all');
        cost_vals = [cost_vals cost_val];
        params = cat(2,params,Cnd_Val);
        params_2 = cat(2,params_2,Cnd_Val_apro);
    end
    
    % save t to be used later
    if mod(k,10) == 0 || k==1
        t0 = t;
    end

    % update t
    s = pagefun(@mtimes,PT_ort,(t-d)) + Tpinvy;
    % for mi=1:M, s(:,:,mi) = PT_ort(:,:,mi)*(t(:,:,mi)-d(:,:,mi)) + Tpinvy(:,:,mi); end
%     s = 0.5*s+0.5*t; % over relaxation
    % update s
    t = max(soft_thresh(s+d,c1rho),c2);
    % update the dual variables
    d = d + s-t;
    
    if mod(k,10) == 0 || k==1
        % st = s-t; tt02 = (t-t0).^2;
        % primal feasibility
        res_p = sqrt(sum((s-t).^2,'all'));
        % dual feasibility
        res_d = sqrt(sum((rho.*Rhov.*(t-t0)).^2,'all'));
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if (mod(k,10) == 0 || k==1) && k<500
%         st = s-t; tt0 = t-t0;
        % st2 = st.^2;
        % primal feasibility
        res_pv = sqrt(pagefun(@mtimes,ones1NNcL,(s-t).^2));
        % res_pv = sqrt(sum((s-t).^2,2));
        % dual feasibility
        res_dv = rho.*sqrt(pagefun(@mtimes,pagefun(@transpose,Rhov.^2),(t-t0).^2));
        
        % update rho
        idx = and(res_pv > 10*res_dv,rho<1e5);
        % it doesn't matter 1e5 or 1e10
        if any(idx,'all')
            rho(idx) = rho(idx)*2;
            ones1NyM(:) = 1;
            ones1NyM(idx) = 0.5;
            d = d.*ones1NyM;
        end
        idx2 = and(res_dv > 10*res_pv,rho>1e-5);
        % it doesn't matter 1e5 or 1e10
        if any(idx2,'all')
            rho(idx2) = rho(idx2)/2;
            ones1NyM(:) = 1;
            ones1NyM(idx2) = 2;
            d = d.*ones1NyM;
        end
        c1rho = c1./rho; 
        
        % Rho for different dimension
        % primal feasibility
        res_pv2 = sqrt(pagefun(@mtimes,(s-t).^2,onesNy1));
        % dual feasibility
        res_dv2 = Rhov .* sqrt(pagefun(@mtimes,(t-t0).^2,pagefun(@transpose,rho.^2)));
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<thRconv_b); % edited
        idx4 = and(res_dv2 > 10*res_pv2, Rhov>thRconv_s); % edited
        % idx3 = res_pv2 > 10*res_dv2;
        % idx4 = res_dv2 > 10*res_pv2;
        
        if any(idx3,'all') || any(idx4,'all')
            Rhov_new = Rhov;
            Rhov_new(idx3) = Rhov_new(idx3)*2;
            Rhov_new(idx4) = Rhov_new(idx4)/2;
            if any(Tcond.*max(Rhov_new)./min(Rhov_new) < th_cond,'all')
                % for this application, 1e13 may be more suitable than 1e8.
                % This is because the approximation of the matrix condition
                % doesn't seem to be accurate enough (much higher than the
                % actual condition number).
                Rhov = Rhov_new;
                PinvTt = pagefun(@transpose,T)./Rhov;
                
                % tic;
                TPinvTt = pagefun(@mtimes,T,PinvTt);
                PinvTt_invTPinvTt = pagefun(@mrdivide,PinvTt,TPinvTt);
                Tpinvy =  pagefun(@mtimes,PinvTt_invTPinvTt,y);
                PT_ort = I_NNcL - pagefun(@mtimes,PinvTt_invTPinvTt,T);
                % toc;
                % tic;
                % for mi=1:M, TPinvTt(:,:,mi) = T(:,:,mi)*PinvTt(:,:,mi); end
                % PinvTt_invTPinvTt = pagefun(@mrdivide,PinvTt,TPinvTt);
                % for mi=1:M, Tpinvy(:,:,mi) = PinvTt_invTPinvTt(:,:,mi) * y(:,:,mi); end
                % for mi=1:M, PT_ort(:,:,mi) = I_NNcL - PinvTt_invTPinvTt(:,:,mi)*T(:,:,mi); end
                % toc;
                onesNNcL1M(:) = 1;
                onesNNcL1M(idx3) = 0.5;
                onesNNcL1M(idx4) = 2;
                d = d.*onesNNcL1M;
                % Tcond.*max(Rhov_new)./min(Rhov_new)
                % cond(TPinvTt(:,:,1))
                if isdebug
                    Cnd_Val = ones(1,1,M,precision,'gpuArray');
                    for i=1:M
                        Cnd_Val(i) = cond(TPinvTt(:,:,i)*TPinvTt(:,:,i)');
                    end
                    Cnd_Val_apro = Tcond.*max(Rhov,[],1)./min(Rhov,[],1);
                end
            end
            
        end
                      
        c1rho = c1rho./Rhov;
%         res_p2 = norm(res_pv); res_d2 = norm(res_dv);
    end
    k=k+1;    
end

x = t(1:N,:,:);
z = t(N+1:N+L,:,:);
r = t(N+L+1:N+L*2,:,:)*tau1;
d=rho.*Rhov.*d;

cost_val = [];% sum(abs(y-A*x-C*z)./tau,'all')+sum(abs(lambda_a.*x),'all');
end
