function [ D ] = continuumDictionary_smooth( p )
%[ D ] = continuumDictionary_smooth( p )
%   Contruct a dictionary of continuums for p dimensional signals. Each
%   endmember will be smoothend by mixing.
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

[ D ] = continuumDictionary( p );

mixing_coeff = fspecial('gaussian',13,3);
mixing_coeff = sum(mixing_coeff);

D = nanimfilter(D,mixing_coeff);





end