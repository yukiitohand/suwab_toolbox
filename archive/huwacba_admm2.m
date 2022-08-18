function [x,z,w,C,DA,stdabsorp,res_p,res_d] = huwacba_admm2(A,y,wv,varargin)
% [x,z,res_p,res_d] = huwacba_admm2(A,y,wv,varargin)
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
%  Outputs
%     x: estimated abundances (Na x N)
%     z: estimated concave background (L x N)
%     C: matrix (L x L) for z
%     res_p,res_d: primal and dual residuals

%  HUWACB solves the following convex optimization  problem 
%  
%         minimize    (1/2) ||y-Ax-Cz-D_A*a||^2_F + lambda*||a||_1
%           x,z
%         subject to  X>=0, z(2:L-1,:)>=0, and a>=0
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
% weights for each dimensions
weight = ones([L,1]);
% standard deviation for the absorption features
stdabsorp = 0.01;
%
lambda = 0;
% initialization of X0
x0 = 0;
% initialization of Z0
z0 = 0;
% initialization of w0
w0 = 0;
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
            case 'STDABSORP'
                stdabsorp = varargin{i+1};
            case 'LAMBDA'
                lambda = varargin{i+1};
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
                    error('initial X is  inconsistent with M or Y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end
weight = weight(:);
if length(weight)~=L
    error('The size of weight is not correct.');
end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%C = continuumDictionary(L);
C = concaveOperator(wv);
% Cinv = C\eye(L);
% s_c = vnorms(Cinv,1);
% Cinv = bsxfun(@rdivide,Cinv,s_c);
% C = bsxfun(@times,C,s_c');
% C = Cinv;
C = continuumDictionary(wv);
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for absorption features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[DA] = Gauss_absoprtionDictonary(wv,stdabsorp);
% DA = zeros([length(wv),length(wv)]);
% for i=1:size(DA,2)
%     d = -normpdf(wv/1000,wv(i)/1000,stdabsorp);
%     DA(:,i) = d;
% end
% DA_norm = vnorms(DA,1);
% meanIdx = floor(length(wv)/2);
% DA = DA/DA_norm(meanIdx);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rho = 0.01;
s_c = vnorms(C,1);
C = bsxfun(@rdivide,C,s_c);
if Aisempty
    T = [C DA];
else
    T = [A C DA];
end
weight = weight(:);
T = bsxfun(@times,weight,T);
[V,Sigma] = svd(T'*T);
Sigma = diag(Sigma);
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
        x(x<0) = 0;
    else
        x=x0;
    end
end

if z0 == 0
    z = zeros([L,Ny]);
else
    z=w0;
end

if w0 == 0
    w = zeros([L,Ny]);
else
    w=w0;
end

if ~Aisempty
    s = [x;z;w];
else
    s = [z;w];
end
% augmented variables
t = s;
% dual variables
d = zeros([N+L*2,Ny]);


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol_p = sqrt((L*2+N)*Ny)*tol;
tol_d = sqrt((L*2+N)*Ny)*tol;
k=1;
res_p = inf;
res_d = inf;
change_rho = 0;
idx = [1:N,N+2:N+L-1,N+L+1:N+L*2];
update_rho_active = 1;
while (k <= maxiter) && ((abs (res_p) > tol_p) || (abs (res_d) > tol_d)) 
    % save z to be used later
    if mod(k,10) == 0
        t0 = t;
    end
    
    % update t
    t = s+d;
    t(idx,:) = max(t(idx,:),0);
    t(N+L+1:N+L*2,:) = soft(t(N+L+1:N+L*2,:),lambda/rho);
    
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
w = t(N+L+1:N+L*2,:);
 
end
