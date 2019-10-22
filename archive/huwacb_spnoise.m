function [ XA,XC,sp_noise,res ] = huwacb_spnoise( x,A,Y,varargin )
% [ XA,XC ] = huwacb( A,Y )
%   Hyperspectral Unmixing with adaptive concave background with sparse
%   noise
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%   Optional parameters
%       'BLOCKSIZE': scaler, number of vectors that are thrown to cvx 
%                    solverat one time
%                    (default) 500
%       'GAMMA': scaler, trade-off parameter for sparse noise
%                (default) 0.18
%   Output variables
%       XA: (p x N)-array, estimated abundance matrix
%       XC: (d x N)-array, estimated concave background matrix
%       sp_noise: (d x N)-array, estimated sparse noise
%       res: (1 x N)-array, residuals in 2-norm.

blocksize = 1000;
gamma = 0.20;

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
            case 'GAMMA'
                gamma = varargin{i+1};
%                 if (~isinteger(blocksize))
%                        error('blocksize must be a positive integer.');
%                 end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end


[d,p] = size(A);
N = size(Y,2);

L = zeros([d,d]);
for i=2:d-1
    L(i,i) = -1;
    L(i,i-1) = (x(i+1)-x(i)) / (x(i+1)-x(i-1));
    L(i,i+1) = (x(i)-x(i-1)) / (x(i+1)-x(i-1));
end
L = L(2:end-1,:);

XA = zeros(p,N);
XC = zeros(d,N);

numblocks = ceil(N/blocksize);

for n=1:numblocks
    firstidx = blocksize*(n-1)+1;
    lastidx = min(N,blocksize*n);
    s = lastidx-firstidx+1;
    y = Y(:,firstidx:lastidx);
    cvx_begin quiet
        variable xA(p,s)
        variable xC(d,s)
        variable sp_noise(d,s)
        minimize norm(y-A*xA-xC-sp_noise,'fro') + gamma*norm(sp_noise,1)
        subject to
            xA >= 0
            L*xC <= 0
            xC(1,:) >= 0
            xC(d,:) >= 0
    cvx_end
    res(firstidx:lastidx) = vnorms(y-A*xA-xC-sp_noise,1,2);
    XC(:,firstidx:lastidx) = xC;
    XA(:,firstidx:lastidx) = xA;
%     t =toc;
%     fprintf('%d / %d has finished, %f [s]\n',n,numblocks,toc);
end

end

