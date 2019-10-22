function [ XA,XC,res,notConvergeFlag ] = suwacb_DCNNOMP( x,A,Y,R,K,varargin )
% [ XA,XC ] = suwacb( x,A,Y,varargin )
%   sparse unmixing with adaptive concave background divide and conquer
%   approach based on normalized nonnegative orthogonal matching pursuit.
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%       R: (d x N)-array, residual with the subset A(:,K) of the library
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
        varargin{rootval_idx} = false;
        Rtmp = Yn - XCtmp;
        [ XAtmp2,XCtmp2,r_mintmp2,notConvergeFlagtmp2 ]...
            = suwacb_DCNNOMP( x,A,Yn(:,notConvergeFlagtmp),Rtmp(:,notConvergeFlagtmp),[],varargin{:});
        XAtmp(:,notConvergeFlagtmp) = XAtmp2;
        XCtmp(:,notConvergeFlagtmp) = XCtmp2;
        r_mintmp(notConvergeFlagtmp) = r_mintmp2;
        notConvergeFlagtmp(notConvergeFlagtmp) = notConvergeFlagtmp2;
        
    else
        K_bar = setdiff(1:p,K);
        [ XAAKbar,XCAKbar,rAKbar ] = huwacb(x,A(:,K),A(:,K_bar));
        RAKbar = A(:,K_bar) - A(:,K)*XAAKbar - XCAKbar;
        RAKbarNrmed = normalizevec(RAKbar,1,'normtype',2);
        V = RAKbarNrmed' * R;
        [r_mintmp,i_mintmp] = max(V,[],1);
        idxesA = K_bar(i_mintmp);
        u_idxes = unique(idxesA);
        for k = u_idxes
            fprintf('k=%d, length(K)=%d\n',k,length(K))
            k_idxes = find((idxesA==k));
            [ XAtmpk,XCtmpk,rtmpk ] = huwacb(x,A(:,[K k]),Yn(:,k_idxes));
            XAtmp([K k],k_idxes) = XAtmpk;
            XCtmp(:,k_idxes) = XCtmpk;
            r_mintmp(:,k_idxes) = rtmpk;
            notConvergeFlagtmpk = rtmpk>tol;
            notConvergeFlagtmp(k_idxes) = notConvergeFlagtmpk;
            Rtmpk = Yn(:,k_idxes) - A(:,[K k])*XAtmpk - XCtmpk;
            if (length(K)+1)<maxnum
%                 notConvergeFlagtmpk1Idx = find(notConvergeFlagtmpk);
                [ XAtmpk2,XCtmpk2,rtmpk2,notConvergeFlagtmpk2 ]...
                = suwacb_DCNNOMP( x,A,Yn(:,k_idxes(notConvergeFlagtmpk)),...
                Rtmpk(:,notConvergeFlagtmpk),[K k],varargin{:});
                XAtmp(:,k_idxes(notConvergeFlagtmpk)) = XAtmpk2;
                XCtmp(:,k_idxes(notConvergeFlagtmpk)) = XCtmpk2;
                r_mintmp(:,k_idxes(notConvergeFlagtmpk)) = rtmpk2;
                notConvergeFlagtmp(:,k_idxes(notConvergeFlagtmpk)) = notConvergeFlagtmpk2;
            end
        end
    end
    
    res(firstidx:lastidx) = r_mintmp;
    XA(:,firstidx:lastidx) = XAtmp;
    XC(:,firstidx:lastidx) = XCtmp;
    notConvergeFlag(firstidx:lastidx) = notConvergeFlagtmp; 
    
end

end