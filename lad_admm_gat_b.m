function [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val ] = lad_admm_gat_b( A,y,varargin )
% [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val ] =lad_admm_gat_b( A,y,varargin)
%   perform least absolute deviation using a alternating direction method of
%   multipliers with generalized augmentation terms (ADMM-GAT), the 
%   formulation 'b'.
%
% INPUT Parameters
%   A  : [L(channels) x N(endmembers)] library matrix
%   y  : [N(channels) x Ny(pixels)], observation vector.
%
% OUTPUT parameters
%   x : [N x Ny] estimated abundance matrix
%   r : [L x Ny] residual vector (y-Ax-b)
%   d : [(N+L) x Ny] dual variables
%   rho: [1 x Ny] spectral penalty parameters
%   Rhov: [(N+L) x 1] spectral penalty parameters
%   res_pv: scalar, primary residual
%   res_dv: scalar, dual residual
%   cost_val: scalar, cost value
%
% Optional Parameters
%  ## GENERAL PARAMETERS #-------------------------------------------------
%   'MAXITER': integer, 
%       the maximum number of iteration.
%       (default) 1000
%   'TOL': scalar, 
%       tolerance parameter.
%       (default) 1e-4
%   'VERBOSE': boolean, 
%       whether or not print residuals at each iteration or not.
%       (default) 0
%
%  ## COEFFICIENTS #-------------------------------------------------------
%   'LAMBDA_R': scalar, array, size compatible with [L x Ny]
%       Weighted coefficients for residual vector.
%       (default) 1
%   'LAMBDA_A': sparsity constraint on x, scalar or vector. size should be
%               compatible with [N x Ny]
%               (default) 0
%
%  ## INITIAL VALUES #-----------------------------------------------------
%   'X0': array, [N x Ny]
%       initial x 
%       (default) []
%   'R0': array, [L x Ny]
%       initial r (residual matrix)
%       (default) []
%   'D0': array, [(L+N) x Ny]
%       initial dual variables
%       (default) []
%   'RHO': scalar, array, [ 1 x Ny ]
%       spectral penalty parameter.
%       (default) 0.01
%   'Rhov': scalar, array, [ (N+L) x 1]
%       spectral penalty parameter
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
%   This function solve the following unconstrained minimization problem
%   called least absolute deviation
%
%                   minimize || y-Ax ||_1 + || lambda_a .* x ||_1
%                      x
%
%   In the formulation this problem is converted to
%                   minimize || c1.*t ||_1 + I_{Ts=y}(s) 
%                      x
%                   subject to s-t = 0
%   where       _   _                             _          _ 
%          s = |  x  |,  T = [A I_L], and  c_1 = |  lambda_a  |
%              |_ r _|                           |_ lambda_r _|
%   and Ax+r=y
%   The augmented Lagrangian
%       || c1.*t ||_1 + I_{Ts=y}(s) + rho * d' (s-t) + rho/2 * ||s-t||_2^2
%
%   Note: if y has multiple columns, then rho will be updated independently
%   for each column.
%
%   ==== Update History ===================================================
%   Mar 18th, 2018  Yuki Itoh: Created
%   Oct 04th, 2019  Yuki Itoh: Supports GPU, spectral penalty parameters
%                              are sufficiently safeguarded. Single
%                              precition mode is also supported.
%   Nov 04th, 2019  Yuki Itoh: comments updated.
%   Dec 17th, 2019  Yuki Itoh: lambda_a added.



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargin-length(varargin)) ~= 2
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
NL = N+L;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = 1000;
% display only warnings
verbose = false;
% tolerance for the primal and dual residues
tol = 1e-4;
lambda_r = ones(L,Ny);
lambda_a = 0.0;
% spectral penalty parameter
rho = 0.01 * ones(1,Ny);
Rhov = ones(NL,1);
% intial value
x0 = [];
r0 = [];
d0 = [];

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
            case 'LAMBDA_R'
                lambda_r = varargin{i+1};
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
            case 'RHO'
                rho = varargin{i+1}; 
            case 'RHOV'
                Rhov = varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is inconsistent with A or y');
                end
                if size(x0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(x0,2)~= Ny
                    error('Size of X0 is not valid');
                end
            case 'R0'
                r0 = varargin{i+1};
                if (size(r0,1) ~= L)
                    error('initial r is inconsistent with A or y');
                end
                if size(r0,2)==1
                    r0 = repmat(r0,[1,Ny]);
                elseif size(r0,2)~= Ny
                    error('Size of r0 is not valid');
                end
            case 'D0'
                d0 = varargin{i+1};
                if (size(d0,1) ~= (N+L))
                    error('initial D is inconsistent with A or y');
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
                error('Unrecognized option: %s', varargin{i});
        end
    end
end

if gpu
    gpu_varargin = {'gpuArray'};
    A = gpuArray(A); y = gpuArray(y);
    rho = gpuArray(rho); Rhov = gpuArray(Rhov);
    lambda_r = gpuArray(lambda_r);
    lambda_a = gpuArray(lambda_a);
else
    gpu_varargin = {};
end

if strcmpi(precision,'single')
    rho = single(rho); Rhov = single(Rhov); lambda_r = single(lambda_r);
    lambda_a = single(lambda_a);
end

%%
% Rhov = ones(NL,1);
% some matrix for 
tau1 = 0.1;
K = [A tau1*eye(L,precision,gpu_varargin{:})];
PinvKt = K'./Rhov;
KPinvKt = K*PinvKt;
PinvKt_invKPinvKt = PinvKt / KPinvKt;
PinvKt_invKPinvKt_y = PinvKt_invKPinvKt * y;
I_NL = eye(NL,precision,gpu_varargin{:});
P_ort = I_NL - PinvKt_invKPinvKt*K;

c1 = ones(NL,Ny,precision,gpu_varargin{:});
c1(1:N,:) = lambda_a .* ones([N,Ny],precision,gpu_varargin{:});
c1(N+1:NL,:) = lambda_r .* ones(L,Ny,precision,gpu_varargin{:});
c1 = c1*tau1;
c1rho = c1 ./ rho ./ Rhov;

% clear A

%%
% intialization
if isempty(x0) && isempty(d0)
    s = PinvKt_invKPinvKt_y;
    t = soft_thresh(s ,c1rho);
    d = s-t;
elseif ~isempty(x0) && ~isempty(d0) && isempty(r0)
    if gpu
        x0 = gpuArray(x0); d0 = gpuArray(d0);
    end
    r0 = y-A*x0;
    t = [x0;r0];
    d = d0 ./ rho ./Rhov;
elseif ~isempty(x0) && ~isempty(d0) && ~isempty(r0)
    if gpu
        x0 = gpuArray(x0); d0 = gpuArray(d0); r0 = gpuArray(r0);
    end
    t = [x0;r0];
    d = d0 ./ rho ./Rhov;
end

clear x0 d0 r0

%%
% main loop
% tic
tol_p = sqrt((L)*Ny)*tol;
tol_d = sqrt((L)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1,precision,gpu_varargin{:});
ones1NL = ones(1,NL,precision,gpu_varargin{:});

Kcond = cond(K).^2;
thRconv_s = 1e-10./Kcond;
thRconv_b = 1e+10./Kcond;
% set a safeguard parameter for 
switch lower(precision)
    case 'double'
        th_cond = 1e8;
    case 'single'
        th_cond = 1e4;
end

if isdebug
    cost_vals = [];
    params = [];
    params_2 = [];
    Cnd_Val = cond(KPinvKt,2);
    Cnd_Val_apro = Kcond*max(Rhov)/min(Rhov);
end

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d))
    if isdebug
        cost_val = sum(abs(A*t(1:N,:)-y),'all');
        cost_vals = [cost_vals cost_val];
        params = [params Cnd_Val];
        params_2 = [params_2 Cnd_Val_apro];
    end
    % save t to be used later
    if mod(k,10) == 0
        t0 = t; s0=s;
    elseif k==1
        t0= t; s0 = t;
    end
    
    % update s
    s = P_ort*(t-d) + PinvKt_invKPinvKt_y;
    
    % update r
    t = soft_thresh(s+d,c1rho);    
    
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
        st2 = st.^2; ss0 = s-s0; tt0 = t-t0;
        % primal feasibility
        res_pv = sqrt(ones1NL*st2);
        % dual feasibility
%         res_dv = rho.*sqrt(Rhov'.^2*tt02);
        res_dv = rho.*sqrt(Rhov'.^2*abs(ss0.*tt0));
        % update rho
        idx = and(res_pv > 10*res_dv, rho<1e5);
        if any(idx)
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
        end
        idx2 = res_dv > and(10*res_pv, rho>1e-5);
        if any(idx2)
            rho(idx2) = rho(idx2)/2;
            d(:,idx2) = d(:,idx2)*2;
        end
        c1rho = c1./rho;
        
        % Rho for different dimension
        % primal feasibility
        res_pv2 = sqrt(st2*onesNy1);
        % dual feasibility
        res_dv2 = Rhov .* sqrt(abs(ss0.*tt0)*rho'.^2);
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<thRconv_b);
        % Rhov(idx3) = Rhov(idx3)*2;
        % d(idx3,:) = d(idx3,:)/2;
        idx4 = and(res_dv2 > 10*res_pv2,Rhov>thRconv_s);
        % Rhov(idx4) = Rhov(idx4)/2;
        % d(idx4,:) = d(idx4,:)*2;
        if any(idx3) || any(idx4)
            Rhov_new = Rhov;
            Rhov_new(idx3) = Rhov_new(idx3)*2;
            Rhov_new(idx4) = Rhov_new(idx4)/2;
            if Kcond*max(Rhov_new)/min(Rhov_new) < th_cond
                % first I upper bounded with 1e13, realize 1e-8 shows
                % better results
                % this one 
                % fprintf('yes');
                Rhov = Rhov_new;
                PinvKt = K'./Rhov;
                KPinvKt = K*PinvKt;
                PinvKt_invKPinvKt = PinvKt / KPinvKt;
                PinvKt_invKPinvKt_y = PinvKt_invKPinvKt * y;
                P_ort = I_NL - PinvKt_invKPinvKt*K;
                
                d(idx3,:) = d(idx3,:)/2;
                d(idx4,:) = d(idx4,:)*2;
                if isdebug
                    Cnd_Val = cond(KPinvKt,2);
                    Cnd_Val_apro = Kcond*max(Rhov)/min(Rhov);
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

% reverse the dual variable to non-scaling form.
d = rho .* Rhov .* d;
x = t(1:N,:);
r = t(N+1:NL,:);

if gpu
    [d,x,r,rho,Rhov] = gather(d,x,r,rho,Rhov);
end
cost_val = sum(abs(A*x-y),'all');
end