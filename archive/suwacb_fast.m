function [ XA,XC,res ] = suwacb_fast( A,Y,x,varargin )
% [ XA,XC ] = suwacb_fast( x,A,Y,varargin )
%   sparse unmixing with adaptive concave background
%   Input variables
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%       x: (d x 1)-array, wavelengths for spectral channels
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
maxnum = 3;

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

XA = zeros(p,N);
XC = zeros(d,N);
res = zeros(1,N);


for n=1:N %numblocks
    tic;
%     fprintf('n=%d\n',n);
    y = Y(:,n);
    flg = 1;
    % first iteration without any endmembers
    [ ~,xC,C ] = huwacb_admm2( [],y,x,'MAXITER',3000,'TOL',1e-6,'VERBOSE','no' );
    R = y-C*xC;
    r = vnorms(R,1,2);
%     fprintf('Initial Error: %f\n',r);
    if r<tol
        % finish if the residual is below the tolerance
        flg = 0; XA(:,n) = 0; XC(:,n) = C*xC; res(n) = r;
    end
    idx_sel = []; idx_res = 1:p; iter = 0;
    while flg
%         fprintf('iter=%d\n',iter);
        iter = iter + 1;
        if isempty(idx_sel)
            [~,XCAKbar,C] = huwacb_admm2( [],y,x,'MAXITER',3000,'TOL',1e-6,'VERBOSE','no' );
            RAKbar = A(:,idx_res) - C*XCAKbar;
        else
            [ XAAKbar,XCAKbar,C ] = huwacb_admm2( A(:,idx_sel),A(:,idx_res),x,'MAXITER',3000,'TOL',1e-6,'VERBOSE','no' );
            RAKbar = A(:,idx_res) - A(:,idx_sel)*XAAKbar - C*XCAKbar;
        end
        RAKbarNrmed = normalizevec(RAKbar,1,'normtype',2);
        V = RAKbarNrmed' * R;
%         [r_mintmp,i_mintmp] = max(V,[],1);
%         K_barCandidates = i_mintmp
%         K_barCandidates = idx_res(find(V>abs(0.8*r_mintmp)));
        
        [r_mintmp,i_mintmp] = sort(V,1,'descend');
        K_barCandidates = idx_res(i_mintmp(1:10));
        
        r_min=inf; i_r_min = 0;rList = [];
        for i=K_barCandidates
            Atmp = [A(:,idx_sel) A(:,i)];
            [xA,xC,C] = huwacb_admm2( Atmp,y,x,'MAXITER',3000,'TOL',1e-6,'VERBOSE','no' );
            r = vnorms(y-Atmp*xA-C*xC,1,2);
            rList = [rList r];
            if r<r_min
                r_min = r; i_r_min = i; xA_min = xA; xC_min = xC;
%                 sp_noise_min = sp_noise;
            end
        end
%         fprintf('Iter:%d, Index:%d selected\nError: %f\n',iter,i_r_min,r_min);
        idx_sel = [idx_sel i_r_min];
        idx_res = setdiff(idx_res,i_r_min);
        R = y - A(:,idx_sel)*xA_min - C*xC;
        if r_min<tol
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min; XC(:,n) = C*xC_min;
            res(n) = r_min; % sp_noise = sp_noise_min;
        end
        if length(idx_sel) >= maxnum
            % finish if the residual is below the tolerance
            flg = 0; XA(idx_sel,n) = xA_min; XC(:,n) = C*xC_min;
            res(n) = r_min; % sp_noise = sp_noise_min;
        end
    end
    t=toc;
    fprintf('fin. n=%03d,t=%f\n',n,t);
end

end