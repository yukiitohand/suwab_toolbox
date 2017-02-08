function [ C ] = concaveOperator( x )
% [ C ] = concaveOperator( x )
%   create a concave operator for the wavelength samples wv
%  Inputs:
%     x: x samples
%  Outputs:
%     C: concave operator (L x L)

L = length(x);
C = zeros([L,L]);
for i=2:L-1
    C(i,i) = 1;
    C(i,i-1) = -(x(i+1)-x(i)) / (x(i+1)-x(i-1));
    C(i,i+1) = -(x(i)-x(i-1)) / (x(i+1)-x(i-1));
end
C(1,1) = 1; C(L,L) = 1;
end