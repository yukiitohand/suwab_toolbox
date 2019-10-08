function [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val,Kcond ] = lad_admm_gat_b_batch( A,y,varargin )
% [ x,b,r,cvx_opts ] = lad_admm_gat_b_batch( A,y,varargin)
%   perform least absolute deviation using a generalized alternating direction method of
%   multipliers (ADMM), the formulation 'b'
%     Input Parameters
%       A  : [L(channels) x N(endmembers)] library matrix
%       y  : [N(channels) x Ny(pixels)], observation vector.
% 
%     Optional Parameters
%       Maxiter : integer, the maximum number of iteration.
%                 (default) 1000
%       Tol     : scalar, tolerance parameter. (default) 1e-4
%       Verbose : boolean, whether or not print residuals at each iteration
%                 or not. (default) 0
%       rho     : scalar or 1 x Ny vector, spectral penalty parameter
%       x0      : initial x (default) 0
%       r0      : initial r (default) 0
%       d0      : initial dual variable (default) 0
%      * For x0, r0, and d0, initialze those parameters inside the function
%      if they are set to 0. It is recommended to provide all of them for
%      efficient warm start.
% 
%     Output parameters
%       x : [N x Ny] estimated abundance matrix
%       r : [L x Ny] residual vector (y-Ax-b)
%
%   This function solve the following unconstrained minimization problem
%   called least absolute deviation
%
%                   minimize || y-Ax ||_1
%                      x
%
%   In the formulation this problem is converted to
%                   minimize || c1.*t ||_1 + I_{Ts=y}(s) 
%                      x
%                   subject to s-t = 0
%   where       _   _                             _   _ 
%          s = |  x  |,  T = [A I_L], and  c_1 = |  0  |
%              |_ r _|                           |_ 1 _|
%   and Ax+r=y
%   The augmented Lagrangian
%       || c1.*t ||_1 + I_{Ts=y}(s) + rho * d' (s-t) + rho/2 * ||s-t||_2^2
%
%   Note: if y has multiple columns, then rho will be updated independently
%   for each column.
%
%   ==== Update History ===================================================
%   Mar 18th, 2018  Yuki Itoh: Created



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
[~,N,~] = size(A);
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
rho = 0.01 * ones(1,Ny,M,'gpuArray');
Rhov = ones(NL,1,M,'gpuArray');
% intial value
x0 = [];
r0 = [];
d0 = [];

lambda_r = ones(L,Ny,M,'gpuArray');

precision = 'single';
isdebug = false;

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
            case 'DEBUG'
                isdebug = varargin{i+1};
            case 'KCOND'
                Kcond = varargin{i+1};
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

% some matrix for 
tau1 = 0.1;
K = cat(2,A,repmat(tau1*eye(L,precision,'gpuArray'),[1,1,M]));
PinvKt = pagefun(@transpose, K)./Rhov;
KPinvKt = pagefun(@mtimes, K, PinvKt);
PinvKt_invKPinvKt = pagefun(@mrdivide, PinvKt, KPinvKt);
PinvKt_invKPinvKt_y = pagefun(@mtimes, PinvKt_invKPinvKt, y);
I_NL = eye(NL,precision,'gpuArray');
P_ort = I_NL - pagefun(@mtimes, PinvKt_invKPinvKt, K);

c1 = ones(NL,Ny,M,precision,'gpuArray');
c1(1:N,:,:) = 0;
c1(N+1:N+L,:,:) = lambda_r.*ones(L,Ny,M,'gpuArray')*tau1;
c1rho = c1 ./ rho ./ Rhov;

% clear lambda_r A

%%
% intialization
if isempty(x0) && isempty(d0)
    s = PinvKt_invKPinvKt_y;
    t = soft_thresh(s ,c1rho);
    d = s-t;
elseif ~isempty(x0) && ~isempty(d0) && isempty(r0)
    r0 = pagefun(@mtimes,A,x0)-y;
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
onesNy1 = ones(Ny,1,precision,'gpuArray');
ones1NL = ones(1,NL,precision,'gpuArray');
ones1NyM = ones(1,Ny,M,precision,'gpuArray');
onesNL1M = ones(N+L,1,M,precision,'gpuArray');

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
    Cnd_Val = ones(1,1,M,precision,'gpuArray');
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
    s = pagefun(@mtimes,P_ort,(t-d)) + PinvKt_invKPinvKt_y;
    
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
        res_pv = sqrt(pagefun(@mtimes,ones1NL,(s-t).^2));
        % dual feasibility
%         res_dv = rho.*sqrt(Rhov'.^2*tt02);
        res_dv = rho.*sqrt(pagefun(@mtimes,pagefun(@transpose,Rhov.^2),abs((s-s0).*(t-t0))));
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
        res_pv2 = sqrt(pagefun(@mtimes,(s-t).^2,onesNy1));
        % dual feasibility
        res_dv2 = Rhov .* sqrt(pagefun(@mtimes,abs((s-s0).*(t-t0)),pagefun(@transpose,rho.^2)));
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
                PinvKt = pagefun(@transpose, K)./Rhov;
                KPinvKt = pagefun(@mtimes, K, PinvKt);
                PinvKt_invKPinvKt = pagefun(@mrdivide, PinvKt, KPinvKt);
                PinvKt_invKPinvKt_y = pagefun(@mtimes, PinvKt_invKPinvKt, y);
                P_ort = I_NL - pagefun(@mtimes, PinvKt_invKPinvKt, K);
                
                onesNL1M(:) = 1;
                onesNL1M(idx3) = 0.5;
                onesNL1M(idx4) = 2;
                d = d.*onesNL1M;
                
                if isdebug
                    Cnd_Val = ones(1,1,M,precision,'gpuArray');
                    for i=1:M
                        Cnd_Val(i) = cond(KPinvKt(:,:,i)*KPinvKt(:,:,i)');
                    end
                    Cnd_Val_apro = Kcond.*max(Rhov,[],1)./min(Rhov,[],1);
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

cost_val = []; % sum(abs(A*x-y),'all');
end