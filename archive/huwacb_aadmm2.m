% not used
function [x,z,C,res_p,res_d] = huwacb_aadmm2(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacb_aadmm2(A,y,wv,varargin)
% hyperspectral unmixing with adaptive concave background (HUWACB) via 
% adaptive alternating direction method of multipliers (ADMM)
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
%     'Tol': tolearance (default) 1e-4
%     'Maxiter': maximum number of iterations (default) 1000
%     'VERBOSE': {'yes', 'no'}
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%  Outputs
%     x: estimated abundances (Na x N)
%     z: estimated concave background (L x N)
%     C: matrix (L x L) for z
%     res_p,res_d: primal and dual residuals

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz||^2_F + lambda_a .* ||x||_1
%           x,z
%         subject to  x>=0 and z(2:L-1,:)>=0
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
if ~isvector(wv) || ~isnumeric(wv)
    error('wv must be a numeric vector.');
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
% weights for each dimensions
weight = ones([L,1]);
% sparsity constraint on the library
lambda_a = 0.0;

% nonnegativity constraint on the library
nonneg = ones([N,1]);

% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
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
               
            case 'NONNEG'
                nonneg = varargin{i+1};
                nonneg = nonneg(:);
                if ~isscalar(nonneg)
                    if length(nonneg)~=N
                        error('Size of nonneg is not right');
                    end
                else
                    nonneg = ones([N,1])*nonneg;
                end
            
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is  inconsistent with M or Y');
                end
                if size(x0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                end
            case 'Z0'
                z0 = varargin{i+1};
                if (size(z0,1) ~= L)
                    error('initial Z is  inconsistent with M or Y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
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
C = continuumDictionary(wv);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = 0.01;
s_c = vnorms(C,1);
C = bsxfun(@rdivide,C,s_c);
if Aisempty
    T = [C];
else
    T = [A C];
end
weight = weight(:);
T = bsxfun(@times,weight,T);
% [V,Sigma] = svd(T'*T);
[U,Sigma1,V] = svd(T);
Sigma = zeros([size(T,2),1]);
Sigma(1:size(T,1)) = diag(Sigma1).^2;
Sigmarhoinv = 1./(Sigma + rho);
Q = bsxfun(@times,V,Sigmarhoinv') * V.';
y = bsxfun(@times,weight,y);
ayy = T' * y;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~Aisempty
    if x0 == 0
        s= Q*ayy;
        x = s(1:N,:);
%         x(x<0) = 0;
    else
        x=x0;
    end
end

if z0 == 0
    z = zeros([L,Ny]);
else
    z=z0;
end
if ~Aisempty
    s = [x;z];
else
    s = z;
end
% augmented variables
% t = s;
% dual variables
d = zeros([N+L,Ny]);


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
epsilon_corr = 0.2;
idx = [find(nonneg>0)', N+2:N+L-1];

while (k <= maxiter) && ((abs(res_p) > tol_p) || (abs(res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 0
        t0 = t;
    end
    if (mod(k,2) == 0)
        s0=s; d0=d;
    end
%     fprintf('k=%d\n',k);
    % update t
    t = s+d;
    t(idx,:) = max(t(idx,:),0);
    t(1:N,:) = soft_thresh(t(1:N,:),lambda_a./rho);
    
    % update s
    s = Q * (ayy + rho * (t-d));
    
    % update the dual variables
    d = d + s-t;

    % update mu so to using an adaptive way
    if k==2
        dsk0 = rho*d; 
        dtk0 = (d0 + s0 - t)*rho;
        sk0 = s; tk0 = t;
    else
        if k>3 && (mod(k,2) == 0)
            dt = d0 + s0-t;
            ddt = rho*dt-dtk0; dH = t-tk0; dds = rho*d-dsk0; dG = (s-sk0); 
            dGdds = sum(sum(dG.*dds)); dHddt = sum(sum(dH.*ddt));
            ddsnrm = norm(dds,'fro'); ddtnrm = norm(ddt,'fro'); 
            dHnrm = norm(dH,'fro'); dGnrm=norm(dG,'fro');
            alpha_corr = dHddt/(dHnrm*ddtnrm); beta_corr = dGdds/(dGnrm*ddsnrm);
%             fprintf('alpha_corr:%f,beta_corr:%f\n',alpha_corr,beta_corr);
            alpha_corrGTepsilon_corr = alpha_corr>epsilon_corr;
            betacorrGTepsilon_corr = beta_corr>epsilon_corr;
            
            if ~(alpha_corrGTepsilon_corr || betacorrGTepsilon_corr)
                
            else
                rho_old = rho;
                if alpha_corrGTepsilon_corr
                   alpha_sd = ddtnrm^2 / dHddt;
                   alpha_mg = dHddt / dHnrm^2;
                   if alpha_sd > 2*alpha_mg
                       alpha = alpha_sd - 2*alpha_mg;
                   else
                       alpha = alpha_mg;
                   end
                end
                if betacorrGTepsilon_corr
                   beta_sd = ddsnrm^2 / dGdds;
                   beta_mg = dGdds / dGnrm^2;
                   if beta_sd > 2*beta_mg
                       beta = beta_sd - 2*beta_mg;
                   else
                       beta = beta_mg;
                   end
                end
                if alpha_corrGTepsilon_corr && (~betacorrGTepsilon_corr)
                    rho = alpha;
                elseif (~alpha_corrGTepsilon_corr) && betacorrGTepsilon_corr
                    rho = beta;
                elseif alpha_corrGTepsilon_corr && betacorrGTepsilon_corr
                    rho = sqrt(alpha*beta);
                end
%                 fprintf('k=%d,rho=%f\n',k,rho);
                % update matrix Q
                
                Sigmarhoinv = 1./(Sigma + rho);
                Q = bsxfun(@times,V,Sigmarhoinv') * V.';
                
                d=d*(rho_old/rho);
                
                dsk0 = rho*d; sk0 = s; tk0 = t;
                
                

            end
        end
    end
    
    if mod(k,10) == 0
        % primal residue
        res_p = norm(s-t,'fro');
        % dual residue
        res_d = rho*(norm((t-t0),'fro'));
        if  strcmp(verbose,'yes')
            fprintf(' k = %f, res_p = %f, res_d = %f\n',k,res_p,res_d)
        end
    end
    
    k=k+1;    
end
% toc
if Aisempty
    x = [];
else
    x = t(1:N,:);
end
z = t(N+1:N+L,:);
end
