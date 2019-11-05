function [x,z,C,r,d,rho,Rhov,res_p,res_d,cost_val] = huwacbl1_admm_gat_a(A,y,wv,varargin)
% [x,z,C,r,d,rho,Rhov,res_p,res_d] = huwacbl1_admm_gat_a(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% alternating direction method of multipliers with generalized augmented 
% terms (ADMM-GAT). This function is almost identical to 
% huwacbl1_gadmm_a_v2.m.
% This function is stabilized by using assessing a condition number of the
% matrix to be inverted.
%
%  Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x Ny) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
%
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x Nc) for z
%     r: residual (L x Ny)
%     d: estimated dual variables (N+L x Ny)
%     rho: spectral penalty parameter "rho" at the convergence, [1 Ny]
%     Rhov: spectral penalty parameter "Rhov" at the convergence, [(Na+Nc+L), 1]
%     res_p,res_d: primal and dual residuals for feasibility
%     cost_val: scalar, cost value
%
% OPTIONAL Parameters
%  ## GENERAL PARAMETERS #-------------------------------------------------
%   'TOL': scalar,
%       tolearance (default) 1e-4
%   'MAXITER': integer, 
%       maximum number of iterations (default) 1000
%   'VERBOSE': boolean, {'yes','no'}
%       whether or not to print information during optimzation.
%       (default) false
%
%  ## COEFFICIENTS #-------------------------------------------------------
%   'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%               vector, the length must be equal to "N"
%               (default) 0
%   'LAMBDA_R': scalar, array, size compatible with [LxNy]
%       Weighted coefficients for residual vector.
%       (default) 1
%   'LAMBDA_C': scalar, array, size compatible with [LxNc]
%       sparsity constraints of the backgroudn concave bases.
%       (default) 0
%
%  ## INITIAL VALUES #-----------------------------------------------------
%   'X0': array, [N x Ny]
%       initial x (coefficient matrix for the libray A)
%       (default) []
%   'Z0': array, [Nc x Ny]
%       initial z (coefficient matrix for C)
%       (default) []
%   'R0': array, [L x Ny]
%       initial 'r' (residual matrix)
%       (default) []
%   'D0': array, [(Na+Nc+L) x Ny]
%       initial dual variables (non-scaling form)
%       (default) []
%   'rho': sclar array, [1 x Ny]
%       initial spectral penalty parameter for different samples,
%       (default) 0.01
%   'Rhov': sclar array, [(Na+Nc+L) x 1]
%       initial spectral penalty parameter, for different dimensions. 
%       (default) 1
%
%  ## PROCESSING OPTIONS #-------------------------------------------------
%   'PRECISION': string, {'single','double'}
%       precision for withch the computation is performed.
%       (default) 'double'
%   'GPU': boolean,
%       whether or not to use GPU for using computation
%       (default) false
%   'DEBUG': boolean
%       if true, cost_function and condition of the matrix to be inverted
%       are plotted
%       (default) false
%
% # Note ------------------------------------------------------------------
%  This function solves the following convex optimization problem 
%  
%       minimize    ||lambda_r.*(y-Ax-Cz)||_{1,1} + ||lambda_a .* x||_{1,1}
%         x,z        + ||lambda_c.*z||_{1,1}
%         subject to  x>=0 and z>=c2_z
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
%tic;
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
lambda_r = ones(L,Ny);
lambda_c = zeros(L,Ny);
% spectral penalty parameter
rho = 0.01*ones([1,Ny]);
Rhov = ones(N+L*2,1);

% initialization of X0
x0 = [];
% initialization of Z0
z0 = [];
% initialization of r0
r0 = [];
% initialization of Lagrange multipliers, d0
d0 = [];
% base matrix of concave curvature
C = [];

% Precision
precision = 'double';

% wheter or not to use GPU or not
gpu = false;

% DEBUG mode outputs the figure of the cost function and Matrix condition
% number.
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
                % lambda_a = lambda_a(:);
                % if ~isscalar(lambda_a)
                %    if length(lambda_a)~=N
                %        error('Size of lambda_a is not right');
                %    end
                % end
            case 'LAMBDA_R'
                lambda_r = varargin{i+1};
            case 'LAMBDA_C'
                lambda_c = varargin{i+1};
            case 'RHO'
                rho = varargin{i+1};
                if length(rho) ~= Ny && length(rho) ~= 1
                    error('initial rho is not valid.');
                end
                if isscalar(rho)
                    rho = rho*ones(1,Ny);
                end
            case 'RHOV'
                Rhov= varargin{i+1};
                if length(Rhov) ~= (N+L*2) && length(Rhov) ~= 1
                    error('initial Rhov is not valid.');
                end
                if isscalar(Rhov)
                    Rhov = Rhov*ones(N+L*2,1);
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
            case 'R0'
                r0 = varargin{i+1};
                if size(r0,1) ~= L && size(r0,2) ~= Ny
                    error('Size of r0 is not right');
                end
            
            case 'D0'
                d0 = varargin{i+1};
                if (size(d0,1) ~= (N+L*2))
                    error('initial D is inconsistent with A or Y');
                end
                if size(d0,2)==1
                    d0 = repmat(d0,[1,Ny]);
                elseif size(d0,2)~= Ny
                    error('Size of D0 is not valid');
                end
            case 'PRECISION'
                precision = varargin{i+1};
            case 'DEBUG'
                isdebug = varargin{i+1};
            case 'GPU'
                gpu = varargin{i+1};
            otherwise
                error('Unrecognized option: %s',varargin{i});
        end
    end
end

%toc;
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
if isempty(C)
    C = continuumDictionary(wv);
    % C = continuumDictionary_smooth(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
    C = C*2;
    if strcmpi(precision,'single')
        C = single(C);
    end
end

if gpu
    gpu_varargin = {'gpuArray'};
    A = gpuArray(A); y = gpuArray(y); C = gpuArray(C);
    rho = gpuArray(rho); Rhov = gpuArray(Rhov);
    lambda_r = gpuArray(lambda_r); lambda_a = gpuArray(lambda_a);
    lambda_c = gpuArray(lambda_c);
else
    gpu_varargin = {};
end

if strcmpi(precision,'single')
    rho = single(rho); Rhov = single(Rhov);
    lambda_r = single(lambda_r); lambda_a = single(lambda_a);
    lambda_c = single(lambda_c);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 0.01;
ynorms = vnorms(y,1);
tau=ynorms;
if strcmpi(precision,'single')
    tau=single(tau);
end

tau1 = 0.2;
if Aisempty
    T = [C tau1*eye(L,precision,gpu_varargin{:})];
else
    T = [A C tau1*eye(L,precision,gpu_varargin{:})];
end
I_N2L = eye(N+2*L,precision,gpu_varargin{:});
PinvTt = T'./Rhov;
TPinvTt = T*PinvTt;
PinvTt_invTPinvTt = PinvTt / TPinvTt;
Tpinvy =  PinvTt_invTPinvTt*y;
PT_ort = I_N2L - PinvTt_invTPinvTt*T;
% projection operator
c1 = zeros([N+2*L,Ny],precision,gpu_varargin{:});
c1(1:N,:) = lambda_a.*ones([N,Ny],precision,gpu_varargin{:});
c1(N+1:N+L,:) = lambda_c.*ones([L,1],precision,gpu_varargin{:});
c1(N+L+1:N+L*2,:) = lambda_r.*ones([L,1],precision,gpu_varargin{:})./tau*tau1;
c1rho = c1./rho./Rhov;

c2 = zeros([N+2*L,1],precision,gpu_varargin{:});
c2(N+1) = -inf; c2(N+L) = -inf; c2(N+L+1:N+2*L) = -inf;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(b0)
    z0 = C\b0;
end
if ~Aisempty
    if isempty(x0) && isempty(z0) && isempty(r0) && isempty(d0)
        s = Tpinvy;
        t = max(soft_thresh(s,c1rho),c2);
        d = s-t;
    elseif ~isempty(x0) && ~isempty(z0) && ~isempty(r0) && ~isempty(d0)
        if gpu
            x0 = gpuArray(x0); z0 = gpuArray(z0);
            d0 = gpuArray(d0); r0 = gpuArray(r0);
        end
        d=d0./rho./Rhov;
        t = [x0;z0;r0];
        % t = max(soft_thresh(s+d,c1rho),c2);
        % d = d+s-t;
%     elseif ~isempty(x0) && ~isempty(z0) && ~isempty(r0) && ~isempty(s0)
%         t =[x0;z0;r0];
%         d = s0-t;
        
    else 
        error('not implemented yet. Initialization works with all or nothing.');
    end
else
    error('not implemented yet');
end

clear x0 z0 r0 d0

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
onesNy1 = ones(Ny,1,precision,gpu_varargin{:});
ones1NL2 = ones(1,N+L*2,precision,gpu_varargin{:});

Tcond = cond(T*T');
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
    Cnd_Val = cond(TPinvTt,2);
    Cnd_Val_apro = Tcond*max(Rhov)/min(Rhov);
end



while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    if isdebug
        cost_val = sum(abs(y-A*t(1:N,:)-C*t(N+1:N+L,:))./tau,'all')+sum(abs(lambda_a.*t(1:N,:)),'all');
        cost_vals = [cost_vals cost_val];
        params = [params Cnd_Val];
        params_2 = [params_2 Cnd_Val_apro];
    end
    % save t to be used later
    if mod(k,10) == 0 || k==1
        t0 = t;
    end

    % update t
    s = PT_ort * (t-d) + Tpinvy;
%     s = 0.5*s+0.5*t; % over relaxation
    % update s
    t = max(soft_thresh(s+d,c1rho),c2);
    % update the dual variables
    d = d + s-t;
    
    if mod(k,10) == 0 || k==1
        st = s-t; tt02 = (t-t0).^2;
        % primal feasibility
        res_p = norm(st,'fro');        
        % dual feasibility
        res_d = sqrt(Rhov'.^2*tt02*rho'.^2);
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if (mod(k,10) == 0 || k==1)
%         st = s-t; tt0 = t-t0;
        st2 = st.^2;
        % primal feasibility
        res_pv = sqrt(ones1NL2*st2);
        % dual feasibility
        res_dv = rho.*sqrt(Rhov'.^2*tt02);
%         if k==390
%             k;
%         end
        
        % update rho
        idx = and(res_pv > 10*res_dv,rho<1e5);
        if any(idx)
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
        end
        idx2 = and(res_dv > 10*res_pv,rho>1e-5);
        if any(idx2)
            rho(idx2) = rho(idx2)/2;
            d(:,idx2) = d(:,idx2)*2;
        end
        c1rho = c1./rho; 
        
        % Rho for different dimension
        % primal feasibility
        res_pv2 = sqrt(st2*onesNy1);
        % dual feasibility
        res_dv2 = Rhov .* sqrt(tt02*rho'.^2);
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<thRconv_b); % edited
        %Rhov(idx3) = Rhov(idx3)*2;
        %d(idx3,:) = d(idx3,:)/2;
        idx4 = and(res_dv2 > 10*res_pv2, Rhov>thRconv_s); % edited
        %Rhov(idx4) = Rhov(idx4)/2;
        %d(idx4,:) = d(idx4,:)*2;
        if any(idx3) || any(idx4)
            Rhov_new = Rhov;
            Rhov_new(idx3) = Rhov_new(idx3)*2;
            Rhov_new(idx4) = Rhov_new(idx4)/2;
            if Tcond*max(Rhov_new)/min(Rhov_new) < th_cond
                % for this application, 1e13 may be more suitable than 1e8.
                % This is because the approximation of the matrix condition
                % doesn't seem to be accurate enough (much higher than the
                % actual condition number).
                % fprintf('p');
                Rhov = Rhov_new;
                PinvTt = T'./Rhov;
                TPinvTt = T*PinvTt;
                PinvTt_invTPinvTt = PinvTt / TPinvTt;
                Tpinvy =  PinvTt_invTPinvTt*y;
                PT_ort = I_N2L - PinvTt_invTPinvTt*T;
                
                d(idx3,:) = d(idx3,:)/2;
                d(idx4,:) = d(idx4,:)*2;
                if isdebug
                    Cnd_Val = cond(TPinvTt,2);
                    Cnd_Val_apro = Tcond*max(Rhov)/min(Rhov);
                end
            end
        end
                      
        c1rho = c1rho./Rhov;
%         res_p2 = norm(res_pv); res_d2 = norm(res_dv);
    end
    k=k+1;    
end

if isdebug
    figure; plot(cost_vals);
    yyaxis right; plot(params); hold on; plot(params_2);
end

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
r = t(N+L+1:N+L*2,:)*tau1;
d=rho.*Rhov.*d;

if gpu
    [x,z,C,r,d,rho,Rhov] = gather(x,z,C,r,d,rho,Rhov);
end

cost_val = sum(abs(y-A*x-C*z)./tau,'all')+sum(abs(lambda_a.*x),'all');

end