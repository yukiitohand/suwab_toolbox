function [ D ] = continuumDictionary( p )
%[ D ] = continuumDictionary( p )
%   Contruct a dictionary of continuums for p dimensional signals
%   Inputs
%       p: integer scalar, number of dimesions
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

D = zeros(p);
for i=1:p
    D(1:i,i) = linspace(0,1,i);
    D(i:p,i) = linspace(1,0,p-i+1);
end
D(p,p)=1;



end

