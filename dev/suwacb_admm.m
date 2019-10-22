function [ XA,XC,res ] = suwacb_admm( x,A,Y,varargin )
% [ XA,XC ] = suwacb( x,A,Y,varargin )
%   sparse unmixing with adaptive concave background
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
%       XC: (d x N)-array, estimated concave curve matrix
%       sp_noise: (d x N)-array, estimated sparse noise
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
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
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
XC = zeros(d,N);
res = zeros(1,N);


for n=1:N %numblocks
    tic;
    y = Y(:,n);
    flg = 1;
    % first iteration without any endmembers
    % [ xC,r ] = ConcaveFit( x,y );
    [~,xC,Ctmp] =  huwacb_admm2_stflip([],y,x);
    r = vnorms(y- Ctmp*xC,1,2);
    fprintf('Initial Error: %f\n',r);
    if r<tol
        % finish if the residual is below the tolerance
        flg = 0; XA(:,n) = 0; XC(:,n) = xC; res(n) = r;
    end
    idx_sel = []; idx_res = 1:p; iter = 0;
    while flg
        iter = iter + 1;
        r_min=inf; i_r_min = 0;rList = [];
        for i=idx_res
            Atmp = [A(:,idx_sel) A(:,i)];
%             [xA, xC, sp_noise,r] = huwacb_spnoise( x,Atmp,y,'GAMMA',0.20 );
            % [xA, xC,r] = huwacb( x,Atmp,y );
            [xA,xC,Ctmp] =  huwacb_admm2_stflip(Atmp,y,x);
            r = vnorms(y - Atmp*xA - Ctmp*xC,1,2);
            rList = [rList r];
            if r<r_min
                r_min = r; i_r_min = i; xA_min = xA; xC_min = xC;
%                 sp_noise_min = sp_noise;
            end
        end
        fprintf('Iter:%d, Index:%d selected\nError: %f\n',iter,i_r_min,r_min);
        idx_sel = [idx_sel i_r_min];
        idx_res = setdiff(idx_res,i_r_min);
        if r_min<tol
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min; XC(:,n) = xC_min;
            res(n) = r_min; % sp_noise = sp_noise_min;
        end
        if length(idx_sel) >= maxnum
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min; XC(:,n) = xC_min;
            res(n) = r_min; % sp_noise = sp_noise_min;
        end
    end
    
end

end