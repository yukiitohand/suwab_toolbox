function [ D ] = absorptionDictionary( p )
%[ D ] = continuumDictionary( p )
%   Contruct a dictionary of continuums for p dimensional signals
%   Inputs
%       p: integer scalar, number of dimesions
%   Outputs
%       D: dictionary matrix,[p x p]
%


D = zeros(p);
counter = 1;
for i=1:p-1
    for j=i+2:p
        for k=i+1:j-1
            D(i:k,counter) = linspace(0,-1,k-i+1);
            D(k:j,counter) = linspace(-1,0,j-k+1);
            counter = counter + 1;
        end
    end
end

end

