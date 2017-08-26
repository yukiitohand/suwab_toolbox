function [ x,b,e,ancillaries ] = huwacb_nansplitter_v2(A,y,wv,varargin )
% [ varargout ] = huwacb_nansplitter_v2( at_func,A,y,wv,varargin_inherited )
% perform HUWACB considering ignored values in the image
% "NaN" in "y" is considered as ignored values.
% "NaN" in "A" is not taken into consideration
% 
% Input Parameters
%    A: dictionary matrix, [L x N]
%    y: observed spectral signals [L x Ny]
%    wv: wavelenght vectors [L x 1]
%  Optional Parameters
%    same as the huwacb_l1error_admm2.m
%  Output Parameters
%    x: learned abundances of the library A, [N x Ny]
%    b: learned concave background: [L x Ny]
%    e: learned l1 noise [L x Ny]
%    ancillaries: struct, storing ancillary information for the
%    optimization result. Fields are
%      activeIdxes : the indexes of the samples 
%      activeBands : the bands used for this cluster
%      Ctmp        : concave dictionary
%      z           : learned coeficients

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
        error('mixing matrix M and data set y are inconsistent');
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
% initialization of e0
e0 = 0; e0_valuidx = 0;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})    
            case 'X0'
                x0 = varargin{i+1};
                if (size(x0,1) ~= N)
                    error('initial X is  inconsistent with M or Y');
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
                    error('initial Z is  inconsistent with Y');
                end
                if size(z0,2)==1
                    z0 = repmat(z0,[1,Ny]);
                elseif size(z0,2) ~= Ny
                    error('initial Z is  inconsistent with Y');
                end
                z0_valuidx = i+1;
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
    
%% split data set into groups
hash_list = cell([1,Ny]);
y_isnan = isnan(y);

for n=1:Ny
    hashList{n} = DataHash(y_isnan(:,n)); 
end

grpHashs = {};
grpIdxes = [];
grpPtrns = [];
for n=1:Ny
    gid = find(cellfun(@(x) strcmp(hashList{n},x),grpHashs));
    if isempty(gid)
        grpHashs = [grpHashs hashList{n}];
        grpIdxes{end+1} = [n];
        grpPtrns = [grpPtrns y_isnan(:,n)];
    else
        grpIdxes{gid} = [grpIdxes{gid} n];
    end
end

%% perform splitted execution
if Aisempty
    x = [];
else
    x = nan([N,Ny]);
end

b = nan([L Ny]); e = nan([L,Ny]);
ancillaries = [];

if all(isnan(wv))
    
else
    for gidi = 1:length(grpHashs)
        activeIdxes = grpIdxes{gidi};
        activeBands = ~grpPtrns(:,gidi);
        if mean(activeBands)>0.3 % only perform when more than 30% are good bands.
            ytmp = y(activeBands,activeIdxes);
            Atmp = A(activeBands,:);
            wvtmp = wv(activeBands);
            
            if x0_valuidx
                varargin{x0_valuidx} = x0(activeBands,activeIdxes);
            end
            if z0_valuidx
                varargin{z0_valuidx} = z0(activeBands,activeIdxes);
            end
            if e0_valuidx
                varargin{e0_valuidx} = e0(activeBands,activeIdxes);
            end
            
            [xtmp,ztmp,etmp,Ctmp] = huwacb_l1error_admm2(A,y,wv,varargin{:});
            btmp = Ctmp * ztmp;

            if ~Aisempty
                x(:,activeIdxes) = xtmp;
            end
            
            b(activeBands,activeIdxes) = btmp;
            e(activeBands,activeIdxes) = etmp;
            
            ancillaries(gidi).activeIdxes = activeIdxes;
            ancillaries(gidi).activeBands = activeBands;
            ancillaries(gidi).C = Ctmp;
            ancillaries(gidi).Z = ztmp;
        end
    end
end

end
