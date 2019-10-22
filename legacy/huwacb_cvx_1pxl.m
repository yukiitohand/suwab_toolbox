function [x,z,C] = huwacb_cvx_1pxl(A,y,wv,varargin)
% [x,z,C] = huwacb_cvx_1pxl(A,y,wv,varargin)
% Solve L1-constrained hyperspectral unmixing with adaptive background
% using matlab cvx toolbox
% 
%   Inputs
%     A : dictionary matrix (L x N) where Na is the number of atoms in the
%         library and L is the number of wavelength bands
%         If A is empty, then computation is performed only for C
%     y : observation vector (L x 1) where N is the number of the
%     observations.
%     wv: wavelength samples (L x 1)
% 
%  Optional parameters
%     'VERBOSE' : {'yes', 'no'}
%     'LAMBDA_A': sparsity constraint on x, scalar or vector. If it is
%                 vector, the length must be equal to "N"
%                 (default) 0
%     'C'       : Concave bases C [L x L]. This will be created from 'wv'
%                 if not provided
%     'SOLVER'  : set a global solver for cvx. (default) whatever set as
%                 global.
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
            case 'VERBOSE'
                if strcmp(varargin{i+1},'yes')
                    verbose=true;
                elseif strcmp(varargin{i+1},'no')
                    verbose=false;
                else
                    error('verbose is invalid');
                end
            case 'SOLVER'
                tmp = varargin{i+1};
                cvx_solver tmp
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


% cvx_begin
%     cvx_quiet(~verbose);
%     variable s(NL,Ny);
%     minimize(0.5*sum_square(y - T*s) +  norm(c1*s,1) );
%     subject to
%         c2*s >= k
% cvx_end

% cvx_begin
%     cvx_quiet(~verbose);
%     variable s(NL,Ny);
%     minimize(0.5*sum_square(y - T*s) +  f1*s );
%     subject to
%         c2*s >= k
% cvx_end


% this is fastest as far as I see
cvx_begin
    cvx_quiet(~verbose);
    variable s(NL,Ny);
    minimize(0.5*sum_square(y - T*s) +  f1*s );
    subject to
        c2*s <= k
cvx_end
% 
% H = T'*T;
% f = f1-y'*T;
% yy = 0.5*y'*y;
% 
% cvx_begin
%     cvx_quiet(~verbose);
%     variable s(NL,Ny);
%     minimize(0.5*quad_form(s,H) + f*s + yy  );
%     subject to
%         c2*s >= k
% cvx_end

fprintf('OptVal: %e\n',0.5*sum_square(y - T*s) +  f1*s );
x = s(1:N,:);
z = s(N+1:NL,:);
% fprintf('%e\n',0.5*norm(y-A*x-C*z,'fro').^2 + norm(lambda_a.*x,1) );

end
