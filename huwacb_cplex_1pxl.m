function [x,z,C] = huwacb_cplex_1pxl(A,y,wv,varargin)
% [x,z,C] = huwacb_cplex_1pxl(A,y,wv,varargin)
% Solve L1-constrained hyperspectral unmixing with adaptive background
% using matlab cplex optimizer
%   Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x 1) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
% 
%  Optional parameters
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%  Outputs
%     x: estimated abundances (N x 1)
%     z: estimated concave background (L x 1)
%     C: matrix (L x L) for z

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
if Ny~=1
    error('This function only works with y with one column');
end
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
% update initial values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% display cvx detail or not
verbose = false;
% sparsity constraint on the library
lambda_a = 0.0;
% base matrix of concave curvature
C = 0;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                lambda_a = lambda_a(:);
                if ~isscalar(lambda_a)
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                end
            case 'CONCAVEBASE'
                C = varargin{i+1};
                if any(size(C) ~= [L L])
                    error('CONCAVEBASE is invalid size');
                end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end

if C==0
    C = continuumDictionary(wv);
    s_c = vnorms(C,1);
    C = bsxfun(@rdivide,C,s_c);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = [A C];
c1 = zeros([NL,1]);
c1(1:N) = lambda_a;
f1 = c1';
% c1 = diag(c1);
c2 = -ones([NL,1]);
c2(N+1) = 0; c2(NL) = 0;
c2 = diag(c2);
k = zeros(NL,Ny);
% k2 = zeros(NL-2,Ny);


% x = cplexqcp(H,f,Aineq,bineq) solves the quadratically constrained
% linear/quadratic programming problem min 1/2*x'*H*x + f*x subject to
% Aineq*x <= bineq. If no quadratic objective term exists, set H=[].
H = T'*T;
f = f1-y'*T;
tic;
[s,fval,exitflag,output,lambda] = cplexqcp(H,f,c2,k);
toc;

% cvx_begin
%     cvx_quiet(~verbose);
%     variable s(NL,Ny);
%     minimize(0.5*sum_square(y - T*s) +  norm(c1*s,1) );
%     subject to
%         c2*s >= k
% cvx_end

% fprintf('OptVal: %e\n',0.5*(norm(y-T*s,'fro')).^2 + norm(c1*s,1) );
x = s(1:N,:);
z = s(N+1:NL,:);
% fprintf('%e\n',0.5*norm(y-A*x-C*z,'fro').^2 + norm(lambda_a.*x,1) );

end