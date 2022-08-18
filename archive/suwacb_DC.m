function [ XA,XC,res,notConvergeFlag ] = suwacb_DC( x,A,Y,K,varargin )
% [ XA,XC ] = suwacb( x,A,Y,varargin )
%   sparse unmixing with adaptive concave background divide and conquer
%   approach
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%       K: 1d-array, indices of selected atoms
%   Optional parameters
%       'TOL': scalar, tolerance, (default) 0.044
%       'MAXNUM': scalar, maximal number of endmembers, (default) 3
%       'BLOCKSIZE': scaler, number of vectors that are thrown to cvx 
%                    solverat one time
%                    (default) 1000
%       'ROOT': Boolean, if this is a root node
%               (deafault) false
%   Output variables
%       XA: (p x N)-array, estimated abundance matrix
%       XC: (d x N)-array, estimated concave curve matrix
%       res: (1 x N)-array, residuals
%       notConvergeFlag: (1 x N)-array, flag for insufficient convergence

tol = 0.044;
maxnum = 3;
blocksize = 1000;
root = false;
rootval_idx = 0;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs.');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'TOL'
                tol = varargin{i+1};
                if (tol<=0)
                       error('TOL must be positive.');
                end
            case 'MAXNUM'
                maxnum = varargin{i+1};
                if (maxnum<=0)
                       error('MAXNUM must be positive.');
                end
            case 'BLOCKSIZE'
                blocksize = varargin{i+1};
%                 if (~isinteger(blocksize))
%                        error('blocksize must be a positive integer.');
%                 end
            case 'ROOT'
                root = varargin{i+1};
                rootval_idx = i+1;
%                 if (maxnum<=0)
%                        error('MAXNUM must be positive.');
%                 end
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

[d,p] = size(A);
N = size(Y,2);

C = zeros([d,d]);
for i=2:d-1
    C(i,i) = -1;
    C(i,i-1) = (x(i+1)-x(i)) / (x(i+1)-x(i-1));
    C(i,i+1) = (x(i)-x(i-1)) / (x(i+1)-x(i-1));
end
C = C(2:end-1,:);

XA = zeros(p,N);
XC = zeros(d,N);
res = inf*zeros([1,N]);
notConvergeFlag = false([1,N]);

numblocks = ceil(N/blocksize);

for n=1:numblocks %numblocks
    tic;
    firstidx = blocksize*(n-1)+1;
    lastidx = min(N,blocksize*n);
    s = lastidx-firstidx+1;
    Yn = Y(:,firstidx:lastidx);
    
    
    r_mintmp = inf*ones([1,s]);
    i_mintmp = zeros([1,s]);
    
    XAtmp = zeros(p,s);
    XCtmp = zeros(d,s);
    
    if root
        [ XCtmp,r_mintmp ] = ConcaveFit(x,Yn);
        notConvergeFlagtmp = r_mintmp>tol;
        notConvergeFlag(firstidx:lastidx) = notConvergeFlagtmp;
        varargin{rootval_idx} = false;
        [ XAtmp2,XCtmp2,r_mintmp2,notConvergeFlagtmp2 ]...
            = suwacb_DC( x,A,Yn(:,notConvergeFlagtmp),[],varargin{:});
        XAtmp(:,notConvergeFlagtmp) = XAtmp2;
        XCtmp(:,notConvergeFlagtmp) = XCtmp2;
        r_mintmp(:,notConvergeFlagtmp) = r_mintmp2;
        notConvergeFlagtmp(:,notConvergeFlagtmp) = notConvergeFlagtmp2;
        
    else
        K_bar = setdiff(1:p,K);
        for k = K_bar
            fprintf('k=%d, length(K)=%d\n',k,length(K))
            [ XAk,XCk,r ] = huwacb(x,[A(:,K) A(:,k)],Yn);
            m = (r < r_mintmp);
            % error
            XAtmptmp = zeros(p,s);
            XAtmptmp([K k],m) = XAk(:,m);
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            XAtmp(:,m) = XAtmptmp(:,m);
            XCtmp(:,m) = XCk(:,m);
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            r_mintmp(:,m) = r(:,m);
            i_mintmp(:,m) = k;
        end

        notConvergeFlagtmp = r_mintmp>tol;
        notConvergeFlagtmp1Idx = find(notConvergeFlagtmp);
        if length(K)<maxnum
            idxes = i_mintmp(notConvergeFlagtmp1Idx);
            u_idxes = unique(idxes);
            for k = u_idxes
                d_idxesb = (idxes==k);
                [ XAk,XCk,resk,notConvergeFlagk ] = suwacb_DC( x,A,Yn(:,d_idxesb),[K k],varargin{:});
                XAtmp(:,d_idxesb) = XAk;
                XCtmp(:,d_idxesb) = XCk;
                r_mintmp(:,d_idxesb) = resk;
                notConvergeFlagtmp(notConvergeFlagtmp1Idx(d_idxesb)) = notConvergeFlagk;
            end
        end
    end
    
    res(firstidx:lastidx) = r_mintmp;
    XA(:,firstidx:lastidx) = XAtmp;
    XC(:,firstidx:lastidx) = XCtmp;
    notConvergeFlag(firstidx:lastidx) = notConvergeFlagtmp; 
    
end

end