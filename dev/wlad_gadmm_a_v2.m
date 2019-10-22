function [ x,r,d,rho,Rhov,res_pv,res_dv ] = wlad_gadmm_a_v2( A,y,varargin )
% [ x,b,r,cvx_opts ] = wlad_gadmm_a_v2( A,y,varargin)
%   perform weighted least absolute deviation (WLAD):
%
%      minimize  ||lambda_r.*(y-Ax)||_1
%         x
%
%   using a generalized alternating direction method of multipliers (ADMM).
%   By default, lambda_r = 1, So the default problem is the LAD:
%   Base iteration is based on 'lad_gadmm_a_v2.m', but the initialization
%   is changed. No longer accept 'r0' (augmented variable).
%
%      minimize  ||y-Ax||_1
%         x                                      
%
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
%       Lambda_R: weight coefficients for the residual r. the size needs to
%                 be comaptible with the operation (lambda_r .* r)
%       Rho     : scalar or 1 x Ny vector, spectral penalty parameter
%       Rhov    : scalar or L x 1  vector, spectral penalty parameter for
%                 different constraint
%       x0      : initial x (default) []
%       d0      : initial dual variable (default) []
%      * For X0, R0, and D0, initialze those parameters inside the function
%      if they are set to 0. It is recommended to provide all of them for
%      efficient warm start.
% 
%     Output parameters
%       x : [N x Ny] estimated abundance matrix
%       r : [L x Ny] residual vector (y-Ax-b), augmented variable
%       d : [N x Ny] dual variable
%       rho: [1 x Ny] spectral penalty parameter
%       Rhov: [L x 1] spectral penealty parameter
%       res_pv: primal residual
%       res_dv: dual residual 
%       
%
%
%
%   In the formulation this problem is converted to
%                   minimize || lambda_r.*r ||_1
%                      x
%                   subject to Ax - r = y
%  
%   The augmented Lagrangian given a DIAGONAL matrix P (otherwise proximal
%   algorithm cannot be easily defined for the update of r)
%      || lambda_r.*r ||_1 + rho*d'(Ax-r-y) + rho/2*||P(Ax-r-y)||_2^2
%
%   Note: if y has multiple columns, then rho will be updated independently
%   for each column.
%
%   ==== Update History ===================================================
%   Feb.16th, 2019  Yuki Itoh: Created, inspired from 'lad_gadmm_a_v2.m'



%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mixing matrixsize
N = size(A,2);

% data set size
[L,Ny] = size(y);

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
% weight coefficients
lambda_r = 1;
% spectral penalty parameter
rho = 0.01 * ones(1,Ny);
Rhov = 1 * ones(L,1);
% intial value
x0 = [];
d0 = [];

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
            case 'LAMBDA_R'
                lambda_r = varargin{i+1};
            case 'RHO'
                rho = varargin{i+1};
            case 'RHOV'
                Rhov = varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
            case 'D0'
                d0 = varargin{i+1};
            case 'CHECK_VAL'
                check_val = varargin{i+1};
                validateattributes(check_val,{'numeric','logical'},...
                                           {'binary'},mfilename,'verbose');
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%
% check variables
if isscalar(Rhov), Rhov = Rhov.*ones(N+Nc+L,1); end
if isscalar(rho), rho = rho.*ones(1,Ny); end
if isrow(Rhov), Rhov = Rhov'; end
if iscolumn(rho), rho = rho'; end

if check_val
    validateattributes(A,{'numeric'},{'nrows',L},mfilename,'A');
    validateattributes(maxiter,{'numeric'},{'integer','nonnegative'},mfilename,'maxiter');
    validateattributes(tol,{'numeric'},{'nonnegative'},mfilename,'tol');
    validateattributes(verbose,{'numeric','logical'},{'binary'},mfilename,'verbose');
    if ~isempty(x0)
        validateattributes(x0,{'numeric'},{'size',[N,Ny]},mfilename,'X0');
    end
    if ~isempty(d0)
        validateattributes(d0,{'numeric'},{'size',[L,Ny]},mfilename,'D0');
    end

    % check lambda_r
    try
        lambda_r = lambda_r.*ones([L,Ny]);
    catch
        [nrow,ncol] = size(lambda_r);
        error('the size (%d x %d) of lambda_r seems wrong.',nrow,ncol);
    end
    % check Rhov
    try
        Rhov = Rhov.*ones(L,1);
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
% some pre compuatations
PA = Rhov.*A;
AtPA = A.'* PA;
iAtPAPAt = AtPA \ PA';
c1 = lambda_r.*ones(L,Ny);
c1rho = c1 ./ rho ./Rhov;

%%
% intialization
if isempty(x0) && isempty(d0)
    x = iAtPAPAt*y;
    r = soft_thresh(A*x-y ,c1rho);
    d = A*x-r-y;
elseif ~isempty(x0) && ~isempty(d0)
    x = x0; d=d0./rho./Rhov;
    Ax = A*x;
    r = soft_thresh(Ax-y+d,c1rho); 
    d = d + Ax-r-y;
else
    error('not implemented yet. Initialization works with all or nothing.');
end

%%
% main loop
% tic
tol_p = sqrt((L)*Ny)*tol;
tol_d = sqrt((L)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1);
ones1L = ones(1,L);
% ones1N = ones(1,N);
Asq = sum(A.^2,2);
while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save r to be used later
    if mod(k,10) == 0 || k==1
        r0 = r;
    end
    
    ymd = y-d;
    % update x
    x = iAtPAPAt * (ymd+r);
    Ax = A*x;
    % update r   
    r = soft_thresh(Ax-ymd,c1rho); 
    
    % update the dual variables
    d = d + Ax-r-y;
    
    if mod(k,10) == 0 || k==1
        Axmrmy = Ax-r-y; 
        Asqrr0 = Asq.*(r-r0).^2;
        % primal feasibility
        res_p = norm(Axmrmy,'fro');
        % dual feasibility
        res_d = sqrt(Rhov'.^2*Asqrr0*rho'.^2);
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0 || k==1
        Axmrmy2 = Axmrmy.^2;
        % primal feasibility
        res_pv = sqrt(ones1L*Axmrmy2);
        % dual feasibility
        res_dv = rho.*sqrt(Rhov'.^2*Asqrr0);
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
        res_pv2 = sqrt(Axmrmy2*onesNy1);
        % dual feasibility
        res_dv2 = Rhov .* sqrt(Asqrr0*rho'.^2);
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<1e10);
        anyidx3 = any(idx3);
        if anyidx3
            Rhov(idx3) = Rhov(idx3)*2;
            d(idx3,:) = d(idx3,:)/2;
        end
        idx4 = and(res_dv2 > 10*res_pv2,Rhov>1e-10);
        anyidx4 = any(idx4);
        if anyidx4
            Rhov(idx4) = Rhov(idx4)/2;
            d(idx4,:) = d(idx4,:)*2;
        end
        if anyidx3 || anyidx4
            PA = Rhov.*A;
            AtPA = A.' * PA;
            iAtPAPAt = AtPA \ PA';
        end

        c1rho = c1rho./Rhov;
        
    end
    k=k+1;    
end

% reverse the dual variable to non-scaling form.
d = (Rhov*rho) .* d;

end