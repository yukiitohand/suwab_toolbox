function [ x,r,d,rho,Rhov,res_pv,res_dv,cost_val ] = lad_gadmm_b_v2_gpu( A,y,varargin )
% [ x,b,r,cvx_opts ] = lad_gadmm_b( A,y,varargin)
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

A = gpuArray(A);
y = gpuArray(y);

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
% spectral penalty parameter
rho = 0.01 * ones(1,Ny,'gpuArray');
Rhov = ones(NL,1,'gpuArray');
% intial value
x0 = [];
r0 = [];
d0 = [];

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
                rho = gpuArray(rho);
            case 'RHOV'
                Rhov = varargin{i+1};
                Rhov = gpuArray(Rhov);
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
                x0 = gpuArray(x0);
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
                r0 = gpuArray(r0);
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
                d0 = gpuArray(d0);
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%

% some matrix for 
tau1 = 0.1;
K = [A tau1*eye(L,'gpuArray')];
PinvKt = K'./Rhov;
KPinvKt = K*PinvKt;
PinvKt_invKPinvKt = PinvKt / KPinvKt;
PinvKt_invKPinvKt_y = PinvKt_invKPinvKt * y;
P_ort = eye(NL,'gpuArray') - PinvKt_invKPinvKt*K;

c1 = ones(NL,Ny,'gpuArray');
c1(1:N,:) = 0;
c1 = c1*tau1;
c1rho = c1 ./ rho ./ Rhov;

%%
% intialization
if isempty(x0) && isempty(d0)
    s = PinvKt_invKPinvKt_y;
    t = soft_thresh(s ,c1rho);
    d = s-t;
elseif ~isempty(x0) && ~isempty(d0) && isempty(r0)
    r0 = A*x0-y;
    t = [x0;r0];
    d = d0 ./ rho ./Rhov;
elseif ~isempty(x0) && ~isempty(d0) && ~isempty(r0)
    t = [x0;r0];
    d = d0 ./ rho ./Rhov;
end


%%
% main loop
% tic
tol_p = sqrt((L)*Ny)*tol;
tol_d = sqrt((L)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1,'gpuArray');
ones1NL = ones(1,NL,'gpuArray');

Kcond = cond(K*K',1);
thRconv_s = sqrt((1e-3./Kcond));
thRconv_b = sqrt((1e+3./Kcond));

% cost_vals = [];
% params = [];

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d))
%     cost_val = sum(abs(A*t(1:N,:)-y),'all');
%     cost_vals = [cost_vals cost_val];
%     params = [params Kcond*max(Rhov)/min(Rhov)];
    % save r to be used later
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
    if (mod(k,10) == 0 || k==1) %&& k<500
%         st = s-t; tt0 = t-t0;
        st2 = st.^2; ss0 = s-s0; tt0 = t-t0;
        % primal feasibility
        res_pv = sqrt(ones1NL*st2);
        % dual feasibility
%         res_dv = rho.*sqrt(Rhov'.^2*tt02);
        res_dv = rho.*sqrt(Rhov'.^2*abs(ss0.*tt0));
        % update rho
        idx = and(res_pv > 10*res_dv, rho<1e10);
        if any(idx)
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
        end
        idx2 = res_dv > and(10*res_pv, rho>1e-10);
        if any(idx2)
            rho(idx2) = rho(idx2)/2;
            d(:,idx2) = d(:,idx2)*2;
        end
        c1rho = c1./rho; 
        % Rho for different dimension
        
        % Rho for different dimension
        % primal feasibility
        res_pv2 = sqrt(st2*onesNy1);
        % dual feasibility
        res_dv2 = Rhov .* sqrt(abs(ss0.*tt0)*rho'.^2);
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<thRconv_b);
        Rhov(idx3) = Rhov(idx3)*2;
        d(idx3,:) = d(idx3,:)/2;
        idx4 = and(res_dv2 > 10*res_pv2,Rhov>thRconv_s);
        Rhov(idx4) = Rhov(idx4)/2;
        d(idx4,:) = d(idx4,:)*2;
        if any(idx3) || any(idx4)
            PinvKt = K'./Rhov;
            KPinvKt = K*PinvKt;
%                 invKPinvKt = KPinvKt \ eye(L);
            PinvKt_invKPinvKt = PinvKt / KPinvKt;
            PinvKt_invKPinvKt_y = PinvKt_invKPinvKt * y;
            P_ort = eye(NL,'gpuArray') - PinvKt_invKPinvKt*K;
        end
                      
        c1rho = c1rho./Rhov;
%         res_p2 = norm(res_pv); res_d2 = norm(res_dv);
    end
    k=k+1;    
end

% reverse the dual variable to non-scaling form.
d = rho .* Rhov .* d;
x = t(1:N,:);
r = t(N+1:NL,:);

[d,x,r,rho,Rhov] = gather(d,x,r,rho,Rhov);
cost_val = sum(abs(A*x-y),'all');
end