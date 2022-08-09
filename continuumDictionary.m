function [ D ] = continuumDictionary( p,varargin )
%[ D ] = continuumDictionary( p )
%   Contruct a dictionary of continuums for p dimensional signals
%   Inputs
%       p: integer scalar;number of dimesions or, array;x samples
%   Outputs
%       D: dictionary matrix,[p x p]
%
%       ^
%       |------+
%       |     +: +
%       |    + :   +
%       |   +  :     +
%       |  +   :       +
%       | +    :         +
%      -+------------------+----------->
%              i

scale_type = 'uniform';
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'SCALE'
                scale_type = varargin{i+1};
            otherwise
                error('Unrecognized option:, %s', varargin{i});
        end
    end
end


if isnumeric(p) && isvector(p)
    L = length(p);
    if L==1
        D = zeros(p);
        for i=1:p
            D(1:i,i) = linspace(0,1,i);
            D(i:p,i) = linspace(1,0,p-i+1);
        end
        D(p,p)=1;
    elseif L>1
        D = concaveOperator(p);
        Dinv = D\eye(L);
        switch upper(scale_type)
            case 'UNIFORM'
                s_d = vnorms(Dinv,1);
                Dinv = bsxfun(@rdivide,Dinv,s_d);
            case 'NATURAL'

        end
        D = Dinv;
    else
        error('p is empty');
    end
else
    error('p must be a numeric vector (scalar is acceptable).');
end

end

