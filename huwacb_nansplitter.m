function [ varargout ] = huwacb_nansplitter(func,A,y,wv,varargin )
% [ varargout ] = huwacb_nansplitter( at_func,A,y,wv,varargin_inherited )
% perform HUWACB considering ignored values in the image
% "NaN" in "y" is considered as ignored values.
% "NaN" in "A" is not taken into consideration
% 
% Input Parameters
%    func: function handle, below are currently available
%          {@huwacb_admm2,@huwacb_l1error_admm2,@huwacba_admm2}
%    A: dictionary matrix, [L x N]
%    y: observed spectral signals [L x Ny]
%    wv: wavelenght vectors [L x 1]
%  Optional Parameters
%    Depend on the function specified in the input
%  Output Parameters
%        Depend on the function specified in the input
%    @huwacb_admm2: 
%        [x,z,C] See "huwacb_admm2" for detaied dscription.
%    @huwacb_l1error_admm2: 
%        [x,z,e,C] See "huwacb_l1error_admm2" for detaied dscription.    

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
z = nan([L,Ny]);
switch func2str(func)
    case 'huwacb_admm2'

    case 'huwacb_l1error_admm2'
        e=nan([L,Ny]);
    otherwise
        error('function "%s" is not integrated to this function yet.',func2str(func));
end
if all(isnan(wv))
    C = nan([L,L]);
    varargout{1} = x;
    varargout{2} = z;
    varargout{3} = C;
else
    for gidi = 1:length(grpHashs)
        activeIdxes = grpIdxes{gidi};
        activeBands = ~grpPtrns(:,gidi);
        if mean(activeBands)>0.3 % only perform when more than 30% are good bands.
            ytmp = y(activeBands,activeIdxes);
            Atmp = A(activeBands,:);
            wvtmp = wv(activeBands);
            switch func2str(func)
                case 'huwacb_admm2'
                    [xtmp,ztmp,~,~,~] = func(Atmp,ytmp,wvtmp,varargin{:});
                case 'huwacb_l1error_admm2'
                    [xtmp,ztmp,etmp,~,~,~] = func(Atmp,ytmp,wvtmp,varargin{:});
                case 'rhuwacb_admm'

                otherwise
                    error('function "%s" is not integrated to this function yet.',func2str(func));
            end
            if ~Aisempty
                x(:,activeIdxes) = xtmp;
            end
            z(activeBands,activeIdxes) = ztmp;
            switch func2str(func)
                case 'huwacb_admm2'

                case 'huwacb_l1error_admm2'
                    e(activeBands,activeIdxes)=etmp;
                otherwise
                    error('function "%s" is not integrated to this function yet.',func2str(func));
            end
        end
    end

    %% setup output parameters
    C = continuumDictionary(wv);
    varargout{1} = x;
    varargout{2} = z;
    switch func2str(func)
        case 'huwacb_admm2'
            varargout{3}=C;
        case 'huwacb_l1error_admm2'
            varargout{3} = e;
            varargout{4} = C;
        otherwise
            error('function "%s" is not integrated to this function yet.',func2str(func));
    end
end

end
