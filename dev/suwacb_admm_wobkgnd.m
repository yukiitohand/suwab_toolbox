function [ XA,res] = suwacb_admm_wobkgnd( A,Y,varargin )
% [ XA,res] = suwacb_admm_wobkgnd( A,Y,varargin )
%   Sparse unmixing with a greedy algorithm. No background is used.
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%   Optional parameters
%       'TOL': scalar, tolerance, (default) 0.048
%       'MAXNUM': scalar, maximal number of endmembers, (default) 4
%   [future implementation]
%       'BLOCKSIZE': scaler, number of vectors that are thrown to cvx 
%                    solverat one time
%                    (default) 500
%   Output variables
%       XA: (p x N)-array, estimated abundance matrix
%       res: (1 x N)-array, residuals

tol = 0.044;
maxnum = 4;

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
            otherwise
                error('Unrecognized option: %s', varargin{i});
        end
    end
end

[d,p] = size(A);
N = size(Y,2);

% L = zeros([d,d]);
% for i=2:d-1
%     L(i,i) = -1;
%     L(i,i-1) = (x(i+1)-x(i)) / (x(i+1)-x(i-1));
%     L(i,i+1) = (x(i)-x(i-1)) / (x(i+1)-x(i-1));
% end
% L = L(2:end-1,:);

XA = zeros(p,N);
res = zeros(1,N);


for n=1:N %numblocks
    tic;
    y = Y(:,n);
    flg = 1;
    % first iteration without any endmembers
    % Do nothing.
    r = vnorms(y,1,2);
    fprintf('Initial Error: %f\n',r);
    if r<tol
        % finish if the residual is below the tolerance
        flg = 0; XA(:,n) = 0; res(n) = r;
    end
    idx_sel = []; idx_res = 1:p; iter = 0;
    while flg
        iter = iter + 1;
        r_min=inf; i_r_min = 0;rList = [];
        for i=idx_res
            Atmp = [A(:,idx_sel) A(:,i)];
            [xA] = sunsal(Atmp,Y,'POSITIVITY','yes','VERBOSE','no',...
                'ADDONE','no','lambda', 0,'AL_ITERS',2000, 'TOL', 1e-6);
            r = vnorms(y - Atmp*xA );
            rList = [rList r];
            if r<r_min
                r_min = r; i_r_min = i; xA_min = xA;
            end
        end
        fprintf('Iter:%d, Index:%d selected\nError: %f\n',iter,i_r_min,r_min);
        idx_sel = [idx_sel i_r_min];
        idx_res = setdiff(idx_res,i_r_min);
        if r_min<tol
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min;
            res(n) = r_min;
        end
        if length(idx_sel) >= maxnum
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min;
            res(n) = r_min;
        end
    end
    
end

end