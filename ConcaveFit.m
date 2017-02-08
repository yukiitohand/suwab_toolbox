function [ XC,res ] = ConcaveFit( x,Y,varargin )
% [ XC,res ] = ConcaveFit( x,Y,varargin )
%   concave curve fitting on spectra
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       Y: (d x N)-array, each column is an observed spectrum.
%   Optional parameters
%       'BLOCKSIZE': scaler, number of vectors that are thrown to cvx 
%                    solverat one time
%                    (default) 500
%   Output variables
%       XC: (d x N)-array, estimated concave curve matrix
%       res: (1 x N)-array, residuals in 2-norm.

blocksize = 1000;

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
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end


[d,N] = size(Y);

L = zeros([d,d]);
for i=2:d-1
    L(i,i) = -1;
    L(i,i-1) = (x(i+1)-x(i)) / (x(i+1)-x(i-1));
    L(i,i+1) = (x(i)-x(i-1)) / (x(i+1)-x(i-1));
end
L = L(2:end-1,:);

XC = zeros(d,N);
res = zeros(1,N);

numblocks = ceil(N/blocksize);

for n=1:numblocks
    tic;
    firstidx = blocksize*(n-1)+1;
    lastidx = min(N,blocksize*n);
    s = lastidx-firstidx+1;
    y = Y(:,firstidx:lastidx);
    cvx_begin quiet
        variable xC(d,s)
        minimize norm(y-xC,'fro')
        subject to
            L*xC <= 0
            xC(1,:) >= 0
            xC(d,:) >= 0
    cvx_end
    res(firstidx:lastidx) = vnorms(y-xC,1,2);
    XC(:,firstidx:lastidx) = xC;
    t =toc;
    fprintf('%d / %d has finished, %f [s]\n',n,numblocks,toc);
    
end

end