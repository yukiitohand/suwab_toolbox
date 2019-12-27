function [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val,Kcond ] = lad_admm_gat_b_batch_v2( A,y,varargin )
%  [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val,Kcond ]
%   = lad_admm_gat_b_batch( A,y,varargin )
%   perform least absolute deviation using a alternating direction method of
%   multipliers with generalized augmentation terms (ADMM-GAT), the 
%   formulation 'b'.
%
% Input Parameters
%   A  : [L(channels) x N(endmembers) x S] library matrix
%   y  : [N(channels) x Ny(pixels) x S], observation vector.
%
% Output parameters
%   x : [N x Ny x S] estimated abundance matrix
%   r : [L x Ny x S] residual vector (y-Ax-b)
%   d : [(N+L) x Ny x S] dual variables
%   rho: [1 x Ny x S] spectral penalty parameters
%   Rhov: [(N+L) x 1 x S] spectral penalty parameters
%   res_pv: scalar, primary residual
%   res_dv: scalar, dual residual
%   cost_val: scalar, cost value
%   Kcond: [1 x 1 x S] condition number.
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
%   'LAMBDA_R': scalar, array, size compatible with [L x Ny x S]
%       Weighted coefficients for residual vector.
%       (default) 1
%   'LAMBDA_A': sparsity constraint on x, scalar or vector. size should be
%               compatible with [N x Ny x S]
%               (default) 0
%
%  ## INITIAL VALUES #-----------------------------------------------------
%   'X0': array, [N x Ny x S]
%       initial x 
%       (default) []
%   'R0': array, [L x Ny x S]
%       initial r (residual matrix)
%       (default) []
%   'D0': array, [(L+N) x Ny x S]
%       initial dual variables
%       (default) []
%   'RHO': scalar, array, [ 1 x Ny x S ]
%       spectral penalty parameter.
%       (default) 0.01
%   'Rhov': scalar, array, [ (N+L) x 1 x S]
%       spectral penalty parameter
%       (default) 1
% 
%  ## PROCESSING OPTIONS #-------------------------------------------------
%   'PRECISION': string, {'single','double'}
%       precision for withch the computation is performed.
%       (default) 'double'
%   'DEBUG': boolean
%       if true, cost_function and condition of the matrix to be inverted
%       are plotted. probably not working right now.
%       (default) false
% 
% # Note ------------------------------------------------------------------
%
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
%   Nov  4th, 2019  Yuki Itoh: Started to track changes.
%   Dec 17th, 2019  Yuki Itoh: lambda_a added.


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if (nargin-length(varargin)) ~= 2
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
% NL = N+L;

[L,Ny,M] = size(y);
[~,N,MA] = size(A);
NL = N+L;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = int32(1000);
% display only sunsal warnings
verbose = false;
% tolerance for the primal and dual residues
tol = 1e-4;
% spectral penalty parameter
rho = 0.01 * ones(1,Ny,M);
Rhov = ones(NL,1,M);
% intial value
x0 = [];
r0 = [];
d0 = [];

lambda_a = zeros(N,Ny,M);
lambda_r = ones(L,Ny,M);

% wheter or not to use GPU or not
gpu = false;
gpu_out = true;

precision = 'single';
isdebug = false;

Rhov_update = true;

Kcond = [];

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
            case 'RHO'
                rho = varargin{i+1};
            case 'RHOV'
                Rhov = varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
            case 'R0'
                r0 = varargin{i+1};
            case 'D0'
                d0 = varargin{i+1};
            case 'PRECISION'
                precision = varargin{i+1};
            case 'LAMBDA_R'
                lambda_r = varargin{i+1};
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
            case 'DEBUG'
                isdebug = varargin{i+1};
            case 'KCOND'
                Kcond = varargin{i+1};
            case 'RHO_UPDATE'
                Rhov_update = varargin{i+1};
            case 'GPU'
                gpu = varargin{i+1};
            case 'GPU_OUT'
                gpu_out = varargin{i+1};
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
    gpu_varargin = {}; gpu_out = false;
end

if strcmpi(precision,'single')
    rho = single(rho); Rhov = single(Rhov); lambda_r = single(lambda_r);
end

%%

% some matrix for 
tau1 = 0.1;
K = cat(2,A,repmat(tau1*eye(L,precision,gpu_varargin{:}),[1,1,MA]));
I_NL = eye(NL,precision,gpu_varargin{:});
if gpu
    PinvKt = pagefun(@transpose, K)./Rhov;
    KPinvKt = pagefun(@mtimes, K, PinvKt);
    PinvKt_invKPinvKt = pagefun(@mrdivide, PinvKt, KPinvKt);
    PinvKt_invKPinvKt_y = pagefun(@mtimes, PinvKt_invKPinvKt, y);
    P_ort = I_NL - pagefun(@mtimes, PinvKt_invKPinvKt, K);
else
    if MA==1 && all(Rhov==1,'all')
        PinvKt = K'./Rhov(:,:,1);
        KPinvKt = K*PinvKt;
        PinvKt_invKPinvKt = PinvKt / KPinvKt;
        PinvKt_invKPinvKt = repmat(PinvKt_invKPinvKt,[1,1,M]);
    else
        PinvKt = permute(K,[2,1,3])./Rhov;
        KPinvKt = mmx_mkl_multi('mult',K,PinvKt);
        PinvKt_invKPinvKt = zeros(NL,L,M);
        for mi=1:M
            PinvKt_invKPinvKt(:,:,mi) = PinvKt(:,:,mi) / KPinvKt(:,:,mi);
        end
    end
    PinvKt_invKPinvKt_y = mmx_mkl_multi('mult',PinvKt_invKPinvKt,y);
    P_ort = I_NL - mmx_mkl_multi('mult',PinvKt_invKPinvKt,K);
end
c1 = ones(NL,Ny,M,precision,gpu_varargin{:});
c1(1:N,:,:) = lambda_a.*ones(N,Ny,M,gpu_varargin{:});
c1(N+1:N+L,:,:) = lambda_r.*ones(L,Ny,M,gpu_varargin{:})*tau1;
c1rho = c1 ./ rho ./ Rhov;

% clear lambda_r A

%%
% intialization
if isempty(x0) && isempty(d0)
    s = PinvKt_invKPinvKt_y;
    t = soft_thresh(s ,c1rho);
    d = s-t;
elseif ~isempty(x0) && ~isempty(d0) && isempty(r0)
    if gpu
        r0 = pagefun(@mtimes,A,x0)-y;
    else
        r0 = mmx_mkl_multi('mult',A,x0)-y;
    end
    t = cat(1,x0,r0);
    d = d0 ./ rho ./ Rhov;
elseif ~isempty(x0) && ~isempty(d0) && ~isempty(r0)
    t = cat(1,x0,r0);
    d = d0 ./ rho ./ Rhov;
end

clear x0 d0 r0

%%
% main loop
% tic
tol_p = sqrt((L+1)*Ny*M)*tol;
tol_d = sqrt((L+1)*Ny*M)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1,precision,gpu_varargin{:});
ones1NL = ones(1,NL,precision,gpu_varargin{:});
ones1NyM = ones(1,Ny,M,precision,gpu_varargin{:});
onesNL1M = ones(N+L,1,M,precision,gpu_varargin{:});

if isempty(Kcond)
    Kcond = ones(1,1,M,precision); KK = gather(K);
    for i=1:M
        Kcond(i) = cond(KK(:,:,i))^2;
    end
end
thRconv_s = 1e-10./Kcond;
thRconv_b = 1e+10./Kcond;
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
    Cnd_Val = ones(1,1,M,precision,gpu_varargin{:});
    for i=1:M
        Cnd_Val(i) = cond(KPinvKt(:,:,i)*KPinvKt(:,:,i)');
    end
    Cnd_Val_apro = Kcond.*max(Rhov,[],1)./min(Rhov,[],1);
end

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d))
    if isdebug
        cost_val = sum(abs(A*t(1:N,:)-y),'all');
        cost_vals = [cost_vals cost_val];
        params = cat(2,params,Cnd_Val);
        params_2 = cat(2,params_2,Cnd_Val_apro);
    end
    % save r to be used later
    if mod(k,10) == 0
        t0 = t; s0=s;
    elseif k==1
        t0= t; s0 = t;
    end
    
    % update s
    if gpu
        s = pagefun(@mtimes,P_ort,(t-d)) + PinvKt_invKPinvKt_y;
    else
        s = mmx_mkl_multi('mult',P_ort,(t-d)) + PinvKt_invKPinvKt_y;
    end
    % update r
    t = soft_thresh(s+d,c1rho);    
    
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
    if (mod(k,10) == 0 || k==1) %&& k<500
%         st = s-t; tt0 = t-t0;
        % st2 = st.^2; ss0 = s-s0; tt0 = t-t0;
        % primal feasibility
        if gpu
            res_pv = sqrt(pagefun(@mtimes,ones1NL,(s-t).^2));
            % dual feasibility
    %         res_dv = rho.*sqrt(Rhov'.^2*tt02);
            res_dv = rho.*sqrt(pagefun(@mtimes,pagefun(@transpose,Rhov.^2),abs((s-s0).*(t-t0))));
        else
            res_pv = sqrt(mmx_mkl_multi('mult',ones1NL,(s-t).^2));
            res_dv = rho.*sqrt(mmx_mkl_multi('mult',permute(Rhov.^2,[2,1,3]),abs((s-s0).*(t-t0))));
        end
        % update rho
        idx = and(res_pv > 10*res_dv, rho<1e5);
        % it looks this upper bound is improtant for stable convergence. it
        % doesn't matter 1e5, 1e3, or 1e10
        if any(idx,'all')
            rho(idx) = rho(idx)*2;
            ones1NyM(:) = 1;
            ones1NyM(idx) = 0.5;
            d = d.*ones1NyM;
        end
        idx2 = res_dv > and(10*res_pv, rho>1e-5);
        % it looks this lower bound is improtant for stable convergence. it
        % doesn't matter 1e-5, 1e-3, or 1e-10
        if any(idx2,'all')
            rho(idx2) = rho(idx2)/2;
            ones1NyM(:) = 1;
            ones1NyM(idx2) = 2;
            d = d.*ones1NyM;
        end
        c1rho = c1./rho; 
        % Rho for different dimension
        
        % Rho for different dimension
        % primal feasibility
        if Rhov_update
            if gpu
                res_pv2 = sqrt(pagefun(@mtimes,(s-t).^2,onesNy1));
                % dual feasibility
                res_dv2 = Rhov .* sqrt(pagefun(@mtimes,abs((s-s0).*(t-t0)),pagefun(@transpose,rho.^2)));
            else
                res_pv2 = sqrt(mmx_mkl_multi('mult',(s-t).^2,onesNy1));
                res_dv2 = Rhov .* sqrt(mmx_mkl_multi('mult',abs((s-s0).*(t-t0)),permute(rho.^2,[2,1,3])));
            end
            % not sure how much these upper and lower bounding is affecting
            % results.
            idx3 = and(res_pv2 > 10*res_dv2, Rhov<thRconv_b);      
            idx4 = and(res_dv2 > 10*res_pv2, Rhov>thRconv_s);
            % idx3 = res_pv2 > 10*res_dv2;
            % idx4 = res_dv2 > 10*res_pv2;

            if any(idx3,'all') || any(idx4,'all')
                Rhov_new = Rhov;
                Rhov_new(idx3) = Rhov_new(idx3)*2;
                Rhov_new(idx4) = Rhov_new(idx4)/2;

                % if cond(KPinvKt,2) < 1e12
                if any(Kcond.*max(Rhov_new)./min(Rhov_new) < th_cond,'all')
                    % first I upper bounded with 1e13, realize 1e-8 shows
                    % better results
                    % this one 
                    % fprintf('yes');
                    Rhov = Rhov_new;
                    if gpu
                        PinvKt = pagefun(@transpose, K)./Rhov;
                        KPinvKt = pagefun(@mtimes, K, PinvKt);
                        PinvKt_invKPinvKt = pagefun(@mrdivide, PinvKt, KPinvKt);
                        PinvKt_invKPinvKt_y = pagefun(@mtimes, PinvKt_invKPinvKt, y);
                        P_ort = I_NL - pagefun(@mtimes, PinvKt_invKPinvKt, K);
                    else
                        PinvKt = permute(K,[2,1,3])./Rhov;
                        KPinvKt = mmx_mkl_multi('mult',K,PinvKt);
                        for mi=1:M
                            PinvKt_invKPinvKt(:,:,mi) = PinvKt(:,:,mi) / KPinvKt(:,:,mi);
                        end
                        PinvKt_invKPinvKt_y = mmx_mkl_multi('mult',PinvKt_invKPinvKt,y);
                        P_ort = I_NL - mmx_mkl_multi('mult',PinvKt_invKPinvKt,K);
                    end

                    onesNL1M(:) = 1;
                    onesNL1M(idx3) = 0.5;
                    onesNL1M(idx4) = 2;
                    d = d.*onesNL1M;

                    if isdebug
                        Cnd_Val = ones(1,1,M,precision,gpu_varargin{:});
                        for i=1:M
                            Cnd_Val(i) = cond(KPinvKt(:,:,i)*KPinvKt(:,:,i)');
                        end
                        Cnd_Val_apro = Kcond.*max(Rhov,[],1)./min(Rhov,[],1);
                    end

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
x = t(1:N,:,:);
r = t(N+1:NL,:,:);

if gpu
    cost_val = sum(abs(pagefun(@mtimes,A,x)-y)+abs(lambda_a.*x),'all');
else
    cost_val = sum(abs(mmx_mkl_multi('multi',A,x)-y)+abs(lambda_a.*x),'all');
end

if gpu && gpu_out
    [x,r,d,rho,Rhov,res_pv,res_dv,cost_val,Kcond] = gather(x,r,d,rho,Rhov,res_pv,res_dv,cost_val,Kcond);
end


end