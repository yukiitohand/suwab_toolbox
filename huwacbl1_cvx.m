function [ x,b,r,cvx_opts ] = huwacbl1_cvx( A,y,wv,varargin )
% [ x,b,r,cvx_opts ] = huwacbl1_cvx( A,y,wvc,varargin)
%   perform hyperspectral unmixing with adpative concave background using
%   l1-error function.
%     Input Parameters
%       A  : [L(channels) x N(endmembers)] library matrix
%       y  : [L(channels) x Ny(pixels)], observation vector.
%       wvc: [L x 1] wavelength vector
% 
%     Optional Parameters
%       'LAMDA_A' : trade-off parameter for the library matrix A
%                   scalar, vector [N x 1, or 1 x N], 
%                   (or matrix [not implemented yet])
%                   (default) 0
%       'VERBOSE' : {'yes', 'no'}; 
%                 'no' - work silently
%                 'yes' - display warnings
%                  (default) 'no'
%     Output parameters
%       x : [N x Ny] estimated abundance matrix
%       b : [L x Ny] estimated concave background
%       r : [L x Ny] residual vector (y-Ax-b)
%
%--------------------------------------------------------------------------
%   Solve the problem below
%
%       minimize     ||Ax+Cz-y||_1 + ||lambda_a'*x||_1
%          x,z
%       subject to  x >= 0
%                   z(2:L-1,:) >= 0
%   
%  This problem is cast into
%       minimize     || c'u ||_1
%          u
%       subject to  [A C I] * u = y
%                   u([1:N,N+2:N+L-1],:) >= 0

%   where c = [lambda_a; zeros([L,1]) ones([L,1])], u = [x;z;r] and r=Ax+Cz-y

% Author: Yuki Itoh, 2017 (yitoh@engin.umass.edu)



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
        error('mixing matrix A and data set y are inconsistent');
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
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
% display process or not
verbose = 'yes';
% sparsity constraint on the library
lambda_a = 0.0;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_A'
                lambda_a = varargin{i+1};
                if isvector(lambda_a)
                    lambda_a = lambda_a(:);
                    if length(lambda_a)~=N
                        error('Size of lambda_a is not right');
                    end
                elseif isscalar(lambda_a)
                    
                elseif ismatrix(lambda_a)
                    error('not implemented yet');
                else
                    error('lambda_a is not right.');
                end
            case 'VERBOSE'
                verbose = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = concaveOperator(wv);
C = C(2:end-1,:);
x = zeros(N,Ny); b = zeros(L,Ny); r = zeros(L,Ny);
%%
n = N + L * 2;
cl = ones(n,1);
cl(1:N) = lambda_a;
cl(N+1:end) = 1;
% cr = ones(Ny,1);
for i=1:Ny
%     tic;
    tic; [x(:,i),r(:,i),b(:,i)] = huwacbl1_cvx_1pxl(A,y(:,i),C,cl,L,N,n); toc;
%     toc;
end

cvx_opts = [];

end

