function [ x,r,d,rho,res_pv,res_dv ] = lad_admm_a( A,y,varargin )
% [ x,b,r,cvx_opts ] = lad_admm_a( A,y,varargin)
%   perform least absolute deviation using alternating direction method of
%   multipliers (ADMM) 
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
%                   minimize || r ||_1
%                      x
%                   subject to Ax - r = y
%  
%   The augmented Lagrangian
%       || r ||_1 + rho * d' (Ax-r-y) + rho/2 * ||Ax-r-y||_2^2
%
%   Note: if y has multiple columns, then rho will be updated independently
%   for each column.
%
%   ==== Update History ===================================================
%   Mar 15th, 2018  Yuki Itoh: Created



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
% display only sunsal warnings
verbose = false;
% tolerance for the primal and dual residues
tol = 1e-4;
% spectral penalty parameter
rho = 0.01 * ones(1,Ny);
% intial value
x0 = 0;
r0 = 0;
d0 = 0;

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
                if (size(z0,1) ~= L)
                    error('initial r is inconsistent with A or y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                elseif size(z0,2)~= Ny
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
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%
% some matrix for 
AtA = A.'*A;
iAtAAt = AtA \ A';
c1 = ones(1,Ny);
c1rho = c1 ./ rho;

%%
% intialization
if x0==0 % if x0 is not given
    x = iAtAAt*y;
else
    x = x0;
end
if r0==0 % if r0 is not given
    r = soft_thresh(A*x-y ,c1rho);
else
    r = r0;
end
if d0==0 % if d0 is not given
    d = A*x-r-y;
else
    d = d0;
end

%%
% main loop
% tic
tol_p = sqrt((L)*Ny)*tol;
tol_d = sqrt((L)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
% onesNy1 = ones(Ny,1);
ones1L = ones(1,L);

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save r to be used later
    if mod(k,10) == 0 || k==1
        r0 = r;
    end
    
    % update x
    ymd = y-d;
    x = iAtAAt * (ymd+r);
    
    % update r
    Ax = A*x;
    r = soft_thresh(Ax-ymd,c1rho);    
    
    % update the dual variables
    d = d + Ax-r-y;
    
    if mod(k,10) == 0 || k==1
        Axmrmy = Ax-r-y; Armr02 = (A'*(r-r0)).^2;
        % primal feasibility
        res_p = norm(Axmrmy,'fro');
        % dual feasibility
        res_d = norm( rho .* sqrt(sum(Armr02,1)), 'fro');
        if  verbose
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0
        % primal feasibility
        res_pv = sqrt(ones1L*Axmrmy.^2);
        % dual feasibility
        res_dv = rho .* sqrt(sum(Armr02,1));
        
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
    end
    k=k+1;    
end

% reverse the dual variable to non-scaling form.
d = rho .* d;

end