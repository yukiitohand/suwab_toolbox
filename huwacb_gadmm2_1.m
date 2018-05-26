function [x,z,C,d,rhov,res_p,res_d] = huwacb_gadmm2_1(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacb_gadmm2(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% a generalized alternating direction method of multipliers (GADMM)
% sparsity constraint on 
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
%     'D0'      : Initial dual parameters [N+L,L] (non-scaling form)
%                 (default) 0
%     'rho'     : initial spectral penalty parameter, scalar
%                 (default) 0.01
%  Outputs
%     x: estimated abundances (N x Ny)
%     z: estimated concave background (L x Ny)
%     C: matrix (L x L) for z
%     d: estimated dual variables (N+L x Ny)
%     rho: spectral penalty parameter at the convergence 
%     res_p,res_d: primal and dual residuals for feasibility

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz||^2_F + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
%
%  The problem will be converted to a variant of non-negative lasso:
%
%         minimize    (1/2) ||y-Ts||^2_F + ||c1.*s||_1
%           s
%         subject to  s >= c2
%   where       _   _ 
%          s = |  x  |,  T = [A C],
%              |_ z _|                       _       _
%                 _          _              |    0_L  |
%          c_1 = |  lambda_a  |,  and  c2 = |   -inf  | 
%                |_   0_L    _|             | 0_{L-2} |
%                                           |_  -inf _|
%    This is solved via ADMM. The variable splitting used here is:
%         minimize    (1/2) ||y-Ts||^2_F + ||c1.*t||_1 _+ I_{t>=c2}(t)
%           s,t
%         subject to  s-t=0
%
%    The augmented Lagrangian of a generalized ADMM is: 
%         L = (1/2) ||y-Ts||^2_F + ||c1.*t||_1 _+ I_{t>=c2}(t)
%                              + rho*d'(s-t) + 1/2||P(s-t)||^2
%    where P is a diagonal matrix storing spectral penalty parameters.
%
%   Advanced residual balancing is used.
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
% weights for each dimensions
weight = ones([L,1]);
% sparsity constraint on the library
lambda_a = 0.0;
% spectral penalty parameter
rhov = 0.01.*ones([NL,1]);

% % nonnegativity constraint on the library
% nonneg = ones([N,1]);

% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
% initialization of B
b0 = 0;
% initialization of Lagrange multipliers, d0
d0 = 0;
% base matrix of concave curvature
C = 0;

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
            case 'WEIGHT'
                weight = varargin{i+1};
                weight = weight(:);
                if length(weight)~=L
                    error('The size of weight is not correct.');
                end
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end
            case 'RHO'
                rhov = varargin{i+1};
                if length(rhov)==1
                    rhov = rhov.*ones(NL,1);
                elseif length(rhov)~=NL
                    error('The length of RHO is invalid');
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
            case 'B0'
                b0 = varargin{i+1};
                if (size(b0,1) ~= L)
                    error('initial Z is inconsistent with A or Y');
                end
                if size(b0,2)==1
                    b0 = repmat(b0,[1,Ny]);
                elseif size(b0,2)~= Ny
                    error('Size of Z0 is not valid');
                end
            case 'D0'
                d0 = varargin{i+1};
                if (size(d0,1) ~= (N+L))
                    error('initial D is inconsistent with A or Y');
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

if b0~=0 
    if z0~=0
        error('B0 and Z0 are both defined');
    end
end

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
if C==0
    C = continuumDictionary(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rho = 0.01;
if Aisempty
    T = [C];
else
    T = [A C];
%     rhov = 0.01.*ones([NL,1]);
%     rhov = 0.001.*ones([NL,1]);
end
% weight = weight(:);
T = bsxfun(@times,weight,T);

[U,Sigma1,V] = svd(T,'econ');
Pinv = 1./rhov; Pinv_m = diag(Pinv);
Sigma = zeros([L,1]);
Sigma(1:L) = diag(Sigma1).^2;
Sigma_inv_m = diag(1./Sigma);
PinvV = Pinv.*V;
Q = Pinv_m - PinvV * ((Sigma_inv_m + V'*PinvV) \ PinvV');
% Q = ( T'*T+diag(rhov) ) \ eye(NL);
ayy = T' * y;

lambda_a_v = lambda_a.*ones([N,1]);
lambda_a_v_rho = lambda_a_v ./ rhov(1:N);
kappa = zeros([NL,1]);
kappa(1:N) = lambda_a_v_rho;
kappa2 = zeros([NL,1]);
kappa2(N+1) = -inf; kappa2(NL) = -inf;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if b0~=0
    z0 = C\b0;
end

% if ~Aisempty
%     if x0 == 0
%         s= Q*ayy;
%         x = s(1:N,:);
%         x(x<0) = 0;
%     else
%         x=x0;
%     end
% end


if x0==0
%     s = zeros(NL,Ny);
%     if Aisempty
        
        s = Q*ayy;
%         s = max(s,kappa2);
%     else
        
%         rho = rho0;
% %         d = d0*rho;
%         x0 = zeros(N,Ny);
%         d0 = [d0(1:12,:);x0;d0(13:end,:)];
%         x0(1:12,:) = x;
        
%         s = [x;z];
%     end
else
%     s = [x0;z0];
end


% if b0==0
% %     if z0 == 0
% %         z = zeros([L,Ny]);
% %     else
% %         z=z0;
% %     end
% else
%     z0 = C\b0;
% end

% if ~Aisempty
%     s = [x;z];
% else
%     s = z;
% end

% augmented variables
% t = s;
% dual variables
if d0==0
%     d = zeros([N+L,Ny]);
    t = max(s-kappa,kappa2);
    d = s-t;
    
else
    d=d0./rhov;
    t=[x0;z0];
    s = Q * (ayy + rhov.* (t-d));
    t = max(s+d-kappa,kappa2);
    d = d + s-t;
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tic
tol_p = sqrt((L+N)*Ny)*tol;
tol_d = sqrt((L+N)*Ny)*tol;
k=1;
res_p = norm(s-t,'fro');
res_d = norm(rhov.*(t),'fro');
% idx = [1:N, N+2:N+L-1];
onesNy1 = ones(Ny,1);
% ones1NL = ones(1,NL);
%res_pList = res_p; res_dList = res_d;

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 0 || k==1
        s0 = s; t0 = t;
    end
      
%     fprintf('k=%d\n',k);
    
    
    
    % update s
    s = Q * (ayy + rhov .* (t-d));
    % update t
    t = max(s+d-kappa,kappa2);

    % update the dual variables
    d = d + s-t;
    
    if mod(k,10) == 0 || k==1
        % primal feasibility
        st = s-t; ss0 = s-s0; tt0 = t-t0;
        res_p = norm(st,'fro'); %res_pList = [res_pList,res_p];
        % dual feasibility
        res_d = norm(rhov.*(tt0),'fro'); %res_dList = [res_dList, res_d];
%         res_d = norm(rhov'*sqrt(abs(ss0.*tt0)),'fro');
        if  verbose
            fprintf(' k = %f, res_p = %e, res_d = %e\n',k,res_p,res_d)
        end
     end
%     res_p = norm(s-t,'fro'); res_d = norm(rhov.*(s-s0),'fro');
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if ((mod(k,10)==0) || k==1) && k < 300
%     if (mod(k,20) == 0 && k<101) || (mod(k,100)==0 && k>101)
        % primal feasibility
        res_pv = sqrt(st.^2 * onesNy1);
        % dual feasibility
%         res_dv = rhov.*sqrt((tt0).^2 * onesNy1);
        res_dv = rhov.*sqrt(abs(tt0.*ss0) * onesNy1);
        
        idx = res_pv > 10*res_dv;
        if any(idx)
            rhov(idx) = rhov(idx)*2;
            d(idx,:) = d(idx,:)/2;
        end
        idx2 = res_dv > 10*res_pv;
        if any(idx2)
            rhov(idx2) = rhov(idx2)/2;
            d(idx2,:) = d(idx2,:)*2;
        end
        if any(idx) || any(idx2)
            Pinv = 1./rhov; Pinv_m = diag(Pinv);
            PinvV = Pinv.*V;
            % Use Woodbury identity
%             tic;
%             R1 = chol(Sigma_inv_m + V'*PinvV);
%             R1inv = R1\eye(L);
%             Q = Pinv_m - PinvV * (R1inv* R1inv') * PinvV';
%             toc;
%             tic;
            Q = Pinv_m - PinvV * ((Sigma_inv_m + V'*PinvV) \ PinvV');
%             Q = ( T'*T+diag(rhov) ) \ eye(NL);
            kappa(1:N) = lambda_a_v ./ rhov(1:N);
        end
%         end
%         res_p2 = norm(res_pv); res_d2 = norm(res_dv);
        
    end
    k=k+1;    
end

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
d=rhov.*d;
end
