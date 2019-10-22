function [x,z,C,r,d,rho,Rhov,res_p,res_d] = huwacbwl1_gadmm_a_v2(A,y,wv,varargin)
% [x,z,C,r,d,rho,Rhov,res_p,res_d] = huwacbl1_gadmm_a(A,y,wv,varargin)
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
%  Main operation is based on "huwacbl1_gadmm_a_v2". "w" indicates its
%  weighted version.
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
%     'Check_val': binary, whether or not to check the validity of inputs
%                  in detail.
%                  (default) false
%     'YNormalize': binary, whether or not to normalize the columns of y
%                   with respect to their L1-norms.
%                   (default) true
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (Nc x Ny)
%     C: matrix (L x L) for z
%     r: residual (L x Ny) (not exactly equal to (y-Ax-Cz), due to practical convergence limitation)
%     d: estimated dual variables ( (N+Nc+L) x Ny )
%     rho: spectral penalty parameter "rho" at the convergence, [1 Ny]
%     Rhov: spectral penalty parameter "Rhov" at the convergence, [L, 1]
%     res_p,res_d: primal and dual residuals for feasibility
%
%   ==== Update History ===================================================
%   Feb.16th, 2019  Yuki Itoh: Created, inspired from 'huwacbl1_gadmm_a_v2.m'
%   Feb.18th, 2019  Yuki Itoh: 'YNormalize' option is added.
%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% library matrix size
Aisempty = isempty(A); N = size(A,2);
% data set size
[L,Ny] = size(y);
if isrow(wv), wv=wv'; end

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
% sparsity constraint on the concave base matrix
lambda_c = 0.0;
% weight coefficient for the model residual
lambda_r = 1.0;
% spectral penalty parameter
rho = 0.01*ones([1,Ny]);
Rhov = []; % this will be set later after Nc is read

% initialization of X0
x0 = [];
% initialization of Z0
z0 = [];
% initialization of B
b0 = [];
% initialization of r0
r0 = [];
% initialization of Lagrange multipliers, d0
d0 = [];
% base matrix of concave curvature
C = [];
% 
c2_z = [];

% y_normalize option
y_normalize = true;
% check input variables
check_val = false;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MAXITER'
                maxiter = varargin{i+1};
            case 'TOL'
                tol = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
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
            case 'CONCAVEBASE'
                C = varargin{i+1};
            case 'C2_Z'
                c2_z = varargin{i+1};
            case 'Z0'
                z0 = varargin{i+1};
            case 'B0'
                b0 = varargin{i+1};
            case 'R0'
                r0 = varargin{i+1};
            case 'D0'
                d0 = varargin{i+1};
            case 'YNORMALIZE'
                y_normalize = varargin{i+1};
            case 'CHECK_VAL'
                check_val = varargin{i+1};
                validateattributes(check_val,{'numeric','logical'},...
                                           {'binary'},mfilename,'verbose');
            otherwise
                error('Undefined option: %s', varargin{i});
        end
    end
end


%toc;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(C)
    C = continuumDictionary(wv);
    % C = continuumDictionary_smooth(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
    C = C*2;
end
Nc = size(C,2);

if isempty(c2_z)
    c2_z = zeros([Nc,1]);
    c2_z(1) = -inf; c2_z(Nc) = -inf;
end

if isempty(Rhov)
    Rhov = ones(N+Nc+L,1);
end

%%
% double check
% if isrow(lambda_a), lambda_a = lambda_a'; end
% if isrow(lambda_c), lambda_c = lambda_c'; end
% if isrow(lambda_r), lambda_r = lambda_r'; end
if isscalar(Rhov), Rhov = Rhov.*ones(N+Nc+L,1); end
if isscalar(rho), rho = rho.*ones(1,Ny); end
if isrow(Rhov), Rhov = Rhov'; end
if iscolumn(rho), rho = rho'; end
if isrow(c2_z), c2_z = c2_z'; end

% further check if check_val
if check_val
    if ~Aisempty
        validateattributes(A,{'numeric'},{'nrows',L},mfilename,'A');
    end
    validateattributes(wv,{'numeric'},{'column','nrows',L},mfilename,'wv');
    validateattributes(C,{'numeric'},{'nrows',L},mfilename,'ConcaveBase');
    validateattributes(maxiter,{'numeric'},{'integer','nonnegative'},mfilename,'maxiter');
    validateattributes(tol,{'numeric'},{'nonnegative'},mfilename,'tol');
    validateattributes(verbose,{'numeric','logical'},{'binary'},mfilename,'verbose');
    validateattributes(y_normalize,{'numeric','logical'},{'binary'},mfilename,'YNormalize');
    if ~isempty(x0)
        validateattributes(x0,{'numeric'},{'size',[N,Ny]},mfilename,'X0');
    end
    if ~isempty(z0)
        validateattributes(z0,{'numeric'},{'size',[Nc,Ny]},mfilename,'Z0');
    end
    if ~isempty(b0)
        validateattributes(b0,{'numeric'},{'size',[L,Ny]},mfilename,'B0');
    end
    if ~isempty(r0)
        validateattributes(r0,{'numeric'},{'size',[L,Ny]},mfilename,'R0');
    end
    if ~isempty(d0)
        validateattributes(d0,{'numeric'},{'size',[N+Nc+L,Ny]},mfilename,'D0');
    end
    if ~isempty(b0) && ~isempty(z0)
        error('B0 and Z0 are both defined');
    end
    % check lambda_a
    try
        lambda_a = lambda_a.*ones([N,Ny]);
    catch
        [nrow,ncol] = size(lambda_a);
        error('the size (%d x %d) of lambda_a seems wrong.',nrow,ncol);
    end
    % check lambda_c
    try
        lambda_c = lambda_c.*ones([Nc,Ny]);
    catch
        [nrow,ncol] = size(lambda_c);
        error('the size (%d x %d) of lambda_c seems wrong.',nrow,ncol);
    end
    % check lambda_r
    try
        lambda_r = lambda_r.*ones([L,Ny]);
    catch
        [nrow,ncol] = size(lambda_r);
        error('the size (%d x %d) of lambda_r seems wrong.',nrow,ncol);
    end
    % check c2_z
    try
        c2_z = c2_z.*ones(Nc,1);
    catch
        [nrow,ncol] = size(c2_z);
        error('the size (%d x %d) of c2_z seems wrong.',nrow,ncol);
    end
    % check Rhov
    try
        Rhov = Rhov.*ones(N+Nc+L,1);
        if size(Rhov,2)>1
            [nrow,ncol] = size(Rhov);
            error('the size (%d x %d) of Rhov seems wrong.',nrow,ncol);
        end
    catch
        [nrow,ncol] = size(Rhov);
        error('the size (%d x %d) of Rhov seems wrong.',nrow,ncol);
    end
    % check rho
    try
        rho = rho.*ones(1,Ny);
        if size(rho,1)>1
            [nrow,ncol] = size(rho);
            error('the size (%d x %d) of Rhov seems wrong.',nrow,ncol);
        end
    catch
        [nrow,ncol] = size(rho);
        error('the size (%d x %d) of rho seems wrong.',nrow,ncol);
    end
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if y_normalize
    ynorms = vnorms(y,1);
    tau=ynorms;
else
    tau = 1;
end

tau1 = 0.2;
if Aisempty
    T = [C tau1*eye(L)];
else
    T = [A C tau1*eye(L)];
end
RhovinvTt = T'./Rhov;
TRhovinvTt = T*RhovinvTt;
Tpinvy = RhovinvTt * (TRhovinvTt \ y);
PT_ort = eye(N+Nc+L) - RhovinvTt * (TRhovinvTt \ T);
% projection operator
c1 = zeros([N+Nc+L,Ny]);
c1(1:N,:) = lambda_a.*ones([N,Ny]);
c1(N+1:N+Nc,:) = lambda_c.*ones([Nc,Ny]);
c1(N+Nc+1:N+Nc+L,:) = lambda_r.*ones([L,Ny])./tau*tau1;
c1rho = c1./rho./Rhov;

c2 = zeros([N+Nc+L,1]);
c2(N+1:N+Nc) = c2_z;
c2(N+Nc+1:N+Nc+L) = -inf;

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
        d=d0./rho./Rhov;
        t = [x0;z0;r0];
    else 
        error('not implemented yet. Initialization works with all or nothing.');
    end
else
    error('not implemented yet');
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
tol_p = sqrt((N+Nc+L)*Ny)*tol;
tol_d = sqrt((N+Nc+L)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1);
ones1NL2 = ones(1,N+Nc+L);

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save t to be used later
    if mod(k,10) == 0 || k==1
        t0 = t;
    end

    % update t
    s = PT_ort * (t-d) + Tpinvy;
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
    if (mod(k,10) == 0 || k==1) && k<500
        st2 = st.^2;
        % primal feasibility
        res_pv = sqrt(ones1NL2*st2);
        % dual feasibility
        res_dv = rho.*sqrt(Rhov'.^2*tt02);
        
        % update rho
        idx = and(res_pv > 10*res_dv,rho<1e10);
        if any(idx)
            rho(idx) = rho(idx)*2;
            d(:,idx) = d(:,idx)/2;
        end
        idx2 = and(res_dv > 10*res_pv,rho>1e-10);
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
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<1e5); % edited
        anyidx3 = any(idx3);
        if anyidx3
            Rhov(idx3) = Rhov(idx3)*2;
            d(idx3,:) = d(idx3,:)/2;
        end
        idx4 = and(res_dv2 > 10*res_pv2, Rhov>1e-5); % edited
        anyidx4 = any(idx4);
        if anyidx4
            Rhov(idx4) = Rhov(idx4)/2;
            d(idx4,:) = d(idx4,:)*2;
        end
        if anyidx3 || anyidx4
            PinvTt = T'./Rhov;
            TPinvTt = T*PinvTt;
            PinvTt_invTPinvTt = PinvTt / TPinvTt;
            Tpinvy =  PinvTt_invTPinvTt*y;
            PT_ort = eye(N+Nc+L) - PinvTt_invTPinvTt*T;
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
z = t(N+1:N+Nc,:);
r = t(N+Nc+1:N+Nc+L,:)*tau1;
d= (Rhov*rho).*d;
end
