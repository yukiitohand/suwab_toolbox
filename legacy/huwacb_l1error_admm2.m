function [x,z,e,C,res_p,res_d] = huwacb_l1error_admm2(A,y,wv,varargin)
% [x,z,e,C,res_p,res_d] = huwacb_l1error_admm2(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% alternating direction method of multipliers (ADMM)
%
%  Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x Ny) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
%  Optional parameters
%     'Tol': tolearance (default) 1e-4
%     'Maxiter': maximum number of iterations (default) 1000
%     'VERBOSE': {'yes', 'no'}
%     'LAMBDA_A' : trade off parameter for l1 constraint on abundances of
%                  the library
%                  (default) 0
%     'LAMBDA_E' : trade off parameter for the l1 error, use "inf" to
%                  decline this option
%                  (default) inf
%  Outputs
%     x: estimated abundances (Na x N)
%     z: estimated concave background (L x N)
%     e: estimated l1-error, in case of lambda_e==inf, zero matrix is
%        returend.
%     C: matrix (L x L) for z
%     res_p,res_d: primal and dual residuals

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz-e||^2_F + lambda*||e||_1 + lambda_a .* ||x||_1
%           x,z,e
%         subject to  X>=0 and z(2:L-1,:)>=0
%  where C is the collection of bases to represent the concave background.
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
wv = wv(:);
Lwv = length(wv);
if (L~=Lwv)
    error('the wavelength samples wv is not correct.');
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the optional parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum number of AL iteration
maxiter = 1000;
% display only sunsal warnings
verbose = 'no';
% tolerance for the primal and dual residues
tol = 1e-4;
% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
% initialization of e0
e0 = 0;
% sparsity constraint on the library
lambda_a = 0.0;
% lambda for sparse (l1) noise
lambda_e = inf;
l1e_cnd = 0;

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
                verbose = varargin{i+1};
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is  inconsistent with M or Y');
                end
                if size(x0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(x0,2) ~= Ny
                    error('initial X is  inconsistent with Y');
                end
            case 'Z0'
                z0 = varargin{i+1};
                if (size(z0,1) ~= L)
                    error('initial Z is  inconsistent with Y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                elseif size(z0,2) ~= Ny
                    error('initial Z is  inconsistent with Y');
                end
            case 'E0'
                e0 = varargin{i+1};
                if (size(e0,1) ~= L)
                    error('initial E is  inconsistent with Y');
                end
                if size(e0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(e0,2) ~= Ny
                    error('initial E is  inconsistent with Y');
                end
            case 'LAMBDA_E'
                l1e_cnd = 1;
                lambda_e = varargin{i+1};
                if ~isscalar(lambda_e)
                    if size(lambda_e,2)~=Ny
                        error('Size of lambda_e is not right');
                    end
                    if size(lambda_e,1)~=L
                        error('Size of lambda_e is not right');
                    end
                else
                    if isinf(lambda_e)
                        l1e_cnd = 0;
                    end
                end
                
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end
               
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

%%
% define the size of l1 error
if l1e_cnd
    Le = L;
else
    Le = 0;
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
% % C = bsxfun(@times,C,s_c');
% C = Cinv;

C = continuumDictionary(wv);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = 0.01;

s_c = vnorms(C,1);
C = bsxfun(@rdivide,C,s_c);

if Aisempty 
    if ~l1e_cnd
        T = C;
    else
        T = [C eye(L)];
    end
else
    if ~l1e_cnd
        T = [A C];
    else
        T = [A C eye(L)];
    end
end
[U,Sigma1,V] = svd(T);
Sigma = zeros([size(T,2),1]);
Sigma(1:size(T,1)) = diag(Sigma1).^2;
% [V,Sigma] = svd(T'*T);
% Sigma = diag(Sigma);
Sigmarhoinv = 1./(Sigma + rho);
Q = bsxfun(@times,V,Sigmarhoinv') * V.';
ayy = T' * y;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~Aisempty
    if x0 == 0
        s= Q*ayy;
        x = s(1:N,:);
        x(x<0) = 0;
    else
        x=x0;
    end
end

if z0 == 0
    z = zeros([L,Ny]);
else
    z=z0;
end

if l1e_cnd
    if e0 == 0
        e = zeros([L,Ny]);
    else
        e=e0;
    end
else
    e =zeros([L,Ny]);
end

if Aisempty 
    if ~l1e_cnd
        s = z;
    else
        s = [z;e];
    end
else
    if ~l1e_cnd
        s = [x;z];
    else
        s = [x;z;e];
    end
end


% augmented variables
t = s;
% dual variables
d = zeros([N+L+Le,Ny]);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol_p = sqrt((L+N)*Ny)*tol;
tol_d = sqrt((L+N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
change_rho = 0;
idx_nonneg = [1:N,N+2:N+L-1];
if ~Aisempty
    idx_l1A = 1:N;
end
if l1e_cnd
    idx_l1e = N+L+1:N+L+Le;
end
update_rho_active = 1;
while (k <= maxiter) && ((abs (res_p) > tol_p) || (abs (res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 0
        t0 = t;
    end
    
    % update t
    t = s+d;
    t(idx_nonneg,:) = max(t(idx_nonneg,:),0);
    if ~Aisempty
        t(idx_l1A,:) = soft_thresh(t(idx_l1A,:),lambda_a./rho);
    end
    if l1e_cnd
        t(idx_l1e,:) = soft_thresh(t(idx_l1e,:),lambda_e/rho);
    end
    
    % update s
    s = Q * (ayy + rho * (t-d));
    
    % update the dual variables
    d = d + s-t;

    % update mu so to keep primal and dual residuals whithin a factor of 10
    if mod(k,10) == 0
        % primal residue
        res_p = norm(s-t,'fro');
        % dual residue
        res_d = rho*(norm((t-t0),'fro'));
        if  strcmp(verbose,'yes')
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
        if update_rho_active
            % update rho
            if res_p > 10*res_d
                rho = rho*2;
                d = d/2;
                change_rho = 1;
            elseif res_d > 10*res_p
                rho = rho/2;
                d = d*2;
                change_rho = 1;
            end
            if  change_rho
                % Px and Pb
                Sigmarhoinv = 1./(Sigma + rho);
                Q = bsxfun(@times,V,Sigmarhoinv') * V.';
                change_rho = 0;
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
if l1e_cnd
    e = t(N+L+1:N+L*2,:);
end
 
end
