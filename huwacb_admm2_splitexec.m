% perform huwac_admm2 for each pixel independently
function [ X,Z,C,res,res_p,res_d ] = huwacb_admm2_splitexec( A,y,wv,varargin )
% [ X,Z,C,res,res_p,res_d ] = huwacb_admm2_splitexec( A,y,wv,varargin )
%   This function is a wrapper for "huwacb_admm2.m", which splits the 
%   operation into smaller pieces to save memory usage.
%   Input variables
%       x: (L x 1)-array, wavelengths for spectral channels
%       A: (L x N)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       y: (L x Ny)-array, each column is an observed spectrum.
%      wv: wavelength samples (L x 1)
%   Optional parameters
%       'BLOCKSIZE': scaler, number of vectors that are thrown to cvx 
%                    solverat one time
%                    (default) 500
%       
%             'Tol': tolearance
%                    (default) follow the default of "huwacb_admm2.m"
%         'Maxiter': maximum number of iterations
%                    (default) follow the default of "huwacb_admm2.m"
%  Outputs
%        x: estimated abundances (Na x N)
%        z: estimated concave background (L x N)
%        C: matrix (L x L) for z
%        res: (1 x N)-array, residuals in 2-norm.
%        res_p,res_d: primal and dual residuals

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check optional variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blocksize = 1000;
varargin_inherited = {};
% varargin_inherited is used for passing through some optional parameters 
% to huwacb_admm2.m

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs.');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'BLOCKSIZE'
                blocksize = varargin{i+1};
%                 if (~isinteger(blocksize))
%                        error('blocksize must be a positive integer.');
%                 end
            case 'TOL'
                varargin_inherited = [varargin_inherited {'TOL',varargin{i+1}}];
            case 'MAXITER'
                varargin_inherited = [varargin_inherited {'MAXITER',varargin{i+1}}];
            case 'VERBOSE'
                varargin_inherited = [varargin_inherited {'VERBOSE',varargin{i+1}}];
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Aisempty = isempty(A);
if Aisempty
    N = 0;
else
    [LA,N] = size(A);
end
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
% initialization & main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Aisempty
    X = [];
else
    X = zeros(N,Ny);
end
Z = zeros(L,Ny);
res = zeros(1,Ny); res_p = zeros(1,Ny); res_d = zeros(1,Ny);
numblocks = ceil(Ny/blocksize);

for n=1:numblocks
%     tic
    firstidx = blocksize*(n-1)+1;
    lastidx = min(Ny,blocksize*n);
    ytmp = y(:,firstidx:lastidx);
    [x,z,C,res_pt,res_dt] = huwacb_admm2(A,ytmp,wv,varargin_inherited{:});
    if Aisempty
        res(firstidx:lastidx) = vnorms(ytmp-C*z,1,2);
    else
        res(firstidx:lastidx) = vnorms(ytmp-A*x-C*z,1,2);
    end
    if ~Aisempty
        X(:,firstidx:lastidx) = x;
    end
    Z(:,firstidx:lastidx) = z;
    res_p(firstidx:lastidx) = res_pt; res_d(firstidx:lastidx) = res_dt;
%     t =toc;
%     fprintf('%d / %d has finished, %f [s]\n',n,numblocks,toc);
end

end
