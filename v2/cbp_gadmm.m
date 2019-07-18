function [x,z,d,rho,Rhov,res_p,res_d] = cbp_gadmm(G,h,c1,c2,varargin)
% [x,z,d,rho,Rhov,res_p,res_d] = cbp(G,h,c1,c2)
%   solver for constrained basis pursuit (CBP) problem:
%
%    minimize  ||c1.*x||_1
%       x
%    subject to   Gx=h and x>=c2
%
%   using a generalized version of alternating method of multipliers (ADMM)
%   by using generalized spectral penalty parameters
%   
%  
%  INPUTS
%   G : (L x N) - matrix
%   h : (L x Ny) - matrix
%   c1: coefficient for x in the cost function
%       any size that compatible with the operation (c1.*x)
%   c2: any size compatible with the operation (x>=c2)
%
%  OUTPUTS
%    x: solution (variable) (N x Ny)
%    z: solution (augmented variable) (N x Ny)
%    d: optimal dual variable (N x Ny)
%    rho: a spectral Penalty Parameter (1 x Ny)
%    Rhov: a spectral Penalty Parameter (N x 1)
%    res_p: primal residual
%    res_d: dual residual
%
%  OPTIONAL PARAMTERS
%    'X0': initial x (N x Ny) (default) []
%    'D0': initial d (N x Ny) (default) []
%    'rho': initial rho (1 x Ny) or scalar (default) 0.01
%    'RHOV': initial Rhov (N x 1) or scalar (deault) 1
%    'maxiter': scalar (default) 1000
%    'tol': scalar (default) 1e-4
%    'verbose': boolean (default) false

% note
%error('inputParser is still in progress.');

%%
% tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[LG,N] = size(G); [L,Ny] = size(h);
if (LG ~= L)
    error('G and h are inconsistent');
end

% optional parameters
% maximum number of AL iteration
maxiter_default = 1000;
% display only sunsal warnings
verbose_default = false;
% tolerance for the primal and dual residues
tol_default = 1e-4;
% sparsity constraint on the library
% spectral penalty parameter
rho_default = 0.01*ones([1,Ny]);
Rhov_default = ones(N,1);
x0_default = [];
d0_default = [];

p = inputParser;
% p.KeepUnmatched = true;
addOptional(p,'X0',x0_default);
addOptional(p,'D0',d0_default);
addOptional(p,'rho',rho_default);
addOptional(p,'Rhov',Rhov_default);
addOptional(p,'maxiter',maxiter_default);
addOptional(p,'verbose',verbose_default);
addOptional(p,'tol',tol_default);
% addOptional(p,'X0',x0_default,@isnumeric);
% addOptional(p,'D0',d0_default,@isnumeric);
% addOptional(p,'rho',rho_default,@isnumeric);
% addOptional(p,'Rhov',Rhov_default,@isnumeric);
% addOptional(p,'maxiter',maxiter_default,@isscalar);
% addOptional(p,'verbose',verbose_default,@(x) or(islogical(x),or(x==0,x==1)));
% addOptional(p,'tol',tol_default,@isscalar);

parse(p,varargin{:});
x0 = p.Results.X0;
d0 = p.Results.D0;
rho = p.Results.rho;
Rhov = p.Results.Rhov;
maxiter = p.Results.maxiter;
verbose = p.Results.verbose;
tol = p.Results.tol;

% if isvector(Rhov), Rhov = Rhov(:); end
% if isvector(rho), rho = rho(:)'; end

try
    Rhov = Rhov.*ones(N,1);
    if size(Rhov,2)>1
        [nrow,ncol] = size(Rhov);
        error('the size (%d x %d) of Rhov seems wrong.',nrow,ncol);
    end
catch
    [nrow,ncol] = size(Rhov);
    error('the size (%d x %d) of Rhov seems wrong.',nrow,ncol);
end

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

% if ~isempty(x0)
%      validateattributes(x0,{'size',[N,Ny]},mfilename,'X0');
% end
% 
% if ~isempty(d0)
%      validateattributes(d0,{'size',[N,Ny]},mfilename,'D0');
% end
% toc;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RhovinvGt = G'./Rhov;
GRhovinvGt = G*RhovinvGt;
Gpinvy = RhovinvGt * (GRhovinvGt \ h);
PG_ort = eye(N) - RhovinvGt * (GRhovinvGt \ G);
c1rho = c1./rho./Rhov;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(x0) && isempty(d0)
    x = Gpinvy;
    z = max(soft_thresh(x,c1rho),c2);
    d = x-z;
elseif ~isempty(x0) && ~isempty(d0)
    d = d0./rho./Rhov;
    z = x0;
else 
    error('not implemented yet. Initialization works with all or nothing.');
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tol_p = sqrt((N)*Ny)*tol;
tol_d = sqrt((N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
onesNy1 = ones(Ny,1);
ones1NL2 = ones(1,N);

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save t to be used later
    if mod(k,10) == 0 || k==1
        z0 = z;
    end

    % update s
    x = PG_ort * (z-d) + Gpinvy;

    % update t
    z = max(soft_thresh(x+d,c1rho),c2);
    
    % update the dual variables
    d = d + x-z;
    
    if mod(k,10) == 0 || k==1
        xz = x-z; zz02 = (z-z0).^2;
        % primal feasibility
        res_p = norm(xz,'fro');        
        % dual feasibility
        res_d = sqrt(Rhov'.^2*zz02*rho'.^2);
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
    end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if (mod(k,10) == 0 || k==1) && k<500
        xz2 = xz.^2;
        % primal feasibility
        res_pv = sqrt(ones1NL2*xz2);
        % dual feasibility
        res_dv = rho.*sqrt(Rhov'.^2*zz02);
        
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
        res_pv2 = sqrt(xz2*onesNy1);
        % dual feasibility
        res_dv2 = Rhov .* sqrt(zz02*rho'.^2);
        idx3 = and(res_pv2 > 10*res_dv2, Rhov<1e10);
        Rhov(idx3) = Rhov(idx3)*2;
        d(idx3,:) = d(idx3,:)/2;
        idx4 = and(res_dv2 > 10*res_pv2, Rhov>1e-10);
        Rhov(idx4) = Rhov(idx4)/2;
        d(idx4,:) = d(idx4,:)*2;
        if any(idx3) || any(idx4)
            PinvGt = G'./Rhov;
            GPinvGt = G*PinvGt;
            PinvGt_invGPinvGt = PinvGt / GPinvGt;
            Gpinvy =  PinvGt_invGPinvGt*h;
            PG_ort = eye(N) - PinvGt_invGPinvGt*G;
        end        
        c1rho = c1rho./Rhov;
    end
    k=k+1;    
end

d=rho.*Rhov.*d;

end