function [ x,b,e,d,ancillaries ] = nanhuwacb(A,y,wv,groups,varargin )
% [ varargout ] = nanhuwacb( at_func,A,y,wv,groups,varargin_inherited )
% perform HUWACB considering ignored values in the image
% "NaN" in "y" is considered as ignored values.
% "NaN" in "A" is not taken into consideration
% 
% Input Parameters
%    A: dictionary matrix, [L x N]
%    y: observed spectral signals [L x Ny]
%    wv: wavelenght vectors [L x 1]
%    groups: struct, output of 'nan_grouper', if empty, computed in the
%            function
%  Optional Parameters
%    same as the huwacb_l1error_admm2.m
%  Output Parameters
%    x: learned abundances of the library A, [N x Ny]
%    b: learned concave background: [L x Ny]
%    e: learned l1 noise [L x Ny]
%    d: learned dual parameters, [(N+L) x Ny]
%    ancillaries: struct, storing ancillary information for the
%    optimization result. Fields are
%      - activeIdxes : the indexes of the samples 
%      - activeBands : the bands used for this cluster
%      - C           : concave dictionary
%      - Z           : learned coeficients
%      - rho         : spectral penalty parameter when converged

%% check input variables
% mixing matrix size
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

%% several input will be checked
% initialization of X0
x0 = 0; x0_valuidx = 0;
% initialization of Z0
z0 = 0; z0_valuidx = 0;
% initialization of B0
b0 = 0; b0_valuidx = 0;
% initialization of D0
d0 = 0; d0_valuidx = 0;
% initialization of e0
e0 = 0; e0_valuidx = 0;
% trade-off parameter for the l1-noise
lambda_e=[]; lambda_eisempty = true;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})    
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is  inconsistent with A');
                end
                if size(x0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(x0,2) ~= Ny
                    error('initial X is  inconsistent with Y');
                end
                x0_valuidx = i+1;
            case 'Z0'
                z0 = varargin{i+1};
                if (size(z0,1) ~= L)
                    error('initial Z is  inconsistent with A');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                elseif size(z0,2) ~= Ny
                    error('initial Z is  inconsistent with Y');
                end
                z0_valuidx = i+1;
            case 'B0'
                b0 = varargin{i+1};
                if (size(b0,1) ~= L)
                    error('initial D is  inconsistent with A');
                end
                if size(b0,2)==1
                    b0 = repmat(b0,[1,Ny]);
                elseif size(b0,2) ~= Ny
                    error('initial B is  inconsistent with Y');
                end
                b0_valuidx = i+1;
            case 'D0'
                d0 = varargin{i+1};
                if (size(d0,1) ~= N+L)
                    error('initial D is  inconsistent with A');
                end
                if size(d0,2)==1
                    d0 = repmat(d0,[1,Ny]);
                elseif size(d0,2) ~= Ny
                    error('initial D is  inconsistent with Y');
                end
                d0_valuidx = i+1;
            case 'LAMBDA_E'
                lambda_e=varargin{i+1};
                lambda_eisempty = false;
            case 'E0'
                e0 = varargin{i+1};
                if (size(e0,1) ~= L)
                    error('initial E is  inconsistent with Y');
                end
                if size(e0,2)==1
                    x0 = repmat(x0,[1,Ny]);
                elseif size(e0,2) ~= Ny
                    error('initial E is  inconsistent with Y');
                end
                e0_valuidx = i+1;
%             case 'WEIGHT'
%                 weight_huwacb = varargin{i+1};
%                 weight_huwacb = weight_huwacb(:);
%                 if length(weight_huwacb)~=L
%                     error('The size of weight is not correct.');
%                 end

        end
    end
end

% split data set into groups
if isempty(groups)
    groups = nan_grouper(y);
end

%% perform splitted execution
if Aisempty
    x = [];
else
    x = nan([N,Ny]);
end

b = nan([L Ny]); e = nan([L,Ny]); d = nan([L+N Ny]);
ancillaries = [];

if all(isnan(wv))
    
else
    for gidi = 1:length(groups)
        activeIdxes = groups(gidi).idxs;
        activeBands = ~groups(gidi).ptrn;
        if mean(activeBands)>0.3 % only perform when more than 30% are good bands.
            ytmp = y(activeBands,activeIdxes);
            Atmp = A(activeBands,:);
            wvtmp = wv(activeBands);
            
            if x0_valuidx
                varargin{x0_valuidx} = x0(:,activeIdxes);
            end
            if b0_valuidx
                varargin{b0_valuidx} = b0(activeBands,activeIdxes);
            end
            if z0_valuidx
                varargin{z0_valuidx} = z0(activeBands,activeIdxes);
            end
            if d0_valuidx
                d0tmp = d0(:,activeIdxes); d0tmp = d0tmp([true(N,1);activeBands],:);
                varargin{d0_valuidx} = d0tmp;
            end
            if e0_valuidx
                varargin{e0_valuidx} = e0(activeBands,activeIdxes);
            end
            
            if lambda_eisempty
%                 tic;
                [xtmp,ztmp,Ctmp,dtmp,rho] = huwacb_admm2_stflip(Atmp,ytmp,wvtmp,varargin{:}); % fastest
%                 [xtmp,ztmp,Ctmp,dtmp,rho] = huwacb_gadmm2_1(Atmp,ytmp,wvtmp,varargin{:}); 
%                 [xtmp,ztmp,Ctmp,dtmp,rho] = huwacbl1_admm(Atmp,ytmp,wvtmp,varargin{:});
                btmp = Ctmp * ztmp;
            else
                [xtmp,ztmp,etmp,Ctmp] = huwacb_l1error_admm2(Atmp,ytmp,wvtmp,varargin{:});
                btmp = Ctmp * ztmp;
            end

            if ~Aisempty
                x(:,activeIdxes) = xtmp;
            end
            
            b(activeBands,activeIdxes) = btmp;
            d([true(N,1);activeBands],activeIdxes) = dtmp;
            
            if lambda_eisempty
                e(activeBands,activeIdxes) = 0;
            else
                e(activeBands,activeIdxes) = etmp;
            end
            
            ancillaries(gidi).activeIdxes = activeIdxes;
            ancillaries(gidi).activeBands = activeBands;
            ancillaries(gidi).C = Ctmp;
            ancillaries(gidi).Z = ztmp;
            ancillaries(gidi).rho = rho;
            ancillaries(gidi).D = dtmp;
        end
    end
end

end
