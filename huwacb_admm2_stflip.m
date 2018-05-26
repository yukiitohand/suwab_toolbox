% used to be "huwacb_admm2_test.m"

function [x,z,C,d,rho,res_p,res_d] = huwacb_admm2_stflip(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacb_admm2(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) with L1 
% sparsity constraint. via alternating direction method of multipliers 
% (ADMM). This version, the iteration of s and t is flipped. After the dual
% update, first t is updated and then s is updated.
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
%
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
rho = 0.01;

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
                rho = varargin{i+1};
%             case 'NONNEG'
%                 nonneg = varargin{i+1};
%                 nonneg = nonneg(:);
%                 if ~isscalar(nonneg)
%                     if length(nonneg)~=N
%                         error('Size of nonneg is not right');
%                     end
%                 else
%                     nonneg = ones([N,1])*nonneg;
%                 end
            
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
end
% weight = weight(:);
% T = bsxfun(@times,weight,T);
% [V,Sigma] = svd(T'*T);
[U,Sigma1,V] = svd(T);
Sigma = zeros([NL,1]);
Sigma(1:L) = diag(Sigma1).^2;
Sigmarhoinv = 1./(Sigma + rho);
Q = bsxfun(@times,V,Sigmarhoinv') * V.';
% y = bsxfun(@times,weight,y);
ayy = T' * y;

lambda_a_v = lambda_a.*ones([N,1]);
lambda_a_v_rho = lambda_a_v ./ rho;
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
        s = max(s,kappa2);
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
    d=d0/rho;
    t=[x0;z0];
    s = Q * (ayy + rho * (t-d));
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
res_p = inf;
res_d = inf;
change_rho = 0;
% idx = [1:N, N+2:N+L-1];
update_rho_active = 1;


while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save z to be used later
%     if mod(k,2) == 0 || k==1
        s0 = s;
%     end
      
%     fprintf('k=%d\n',k);

    % update t
    
%     t = s+d;
%     t(idx,:) = max(t(idx,:),0);
%     t(1:N,:) = soft_thresh(t(1:N,:),lambda_a_v);
%     t = max(soft_thresh(s+d,kappa),kappa2);
    t = max(s+d-kappa,kappa2);
%     t = 0.6*t +0.4*s; % under-relaxation
    
    % update s
    s = Q * (ayy + rho * (t-d));
    
    % update the dual variables
    d = d + s-t;
    
%     if mod(k,2) == 0 || k==1
        % primal feasibility
%         res_p = norm(s-t,'fro');
%         % dual feasibility
%         res_d = rho*(norm((s-s0),'fro'));
%         if  verbose
%             fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
%         end
%     end
    
    % update mu so to keep primal and dual feasibility whithin a factor of 10
    if mod(k,10) == 0
        % primal feasibility
        res_p = norm(s-t,'fro');
        % dual feasibility
        res_d = rho*(norm((s-s0),'fro'));
        if  verbose
            fprintf(' k = %f, res_p = %10e, res_d = %10e\n',k,res_p,res_d)
        end
        if update_rho_active
            % update rho
            if res_p > 10*res_d
                rho = rho*3;
                d = d/3;
                change_rho = 1;
            elseif res_d > 10*res_p
                rho = rho/3;
                d = d*3;
                change_rho = 1;
            end
            if  change_rho
                % Px and Pb
                Sigmarhoinv = 1./(Sigma + rho);
                Q = bsxfun(@times,V,Sigmarhoinv') * V.';
                kappa(1:N) = lambda_a_v ./ rho;
                change_rho = 0;
%                 fprintf('k=%d,rho=%d\n',k,rho);
            end
        end
    end
    k=k+1;    
end

if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
d=rho*d;
end
