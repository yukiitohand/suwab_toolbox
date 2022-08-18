function [ C ] = concaveOperator_v2( x )
% [ C ] = concaveOperator( x )
%   create a concave operator for the wavelength samples wv
%  Inputs:
%     x: x samples, L x C
%  Outputs:
%     C: concave operator (L x L x C)

[L,N] = size(x);
% C = zeros([L,L]);

denom = gpuArray(permute(x(3:end,:) - x(1:end-2,:), [1,3,2]));
numer = gpuArray(permute(-x(2:end,:) + x(1:end-1,:),[1,3,2]));
% Cm1 = diag(numer(2:end)./denom);
% Cp1 = diag(numer(1:end-1)./denom);
C = repmat(eye(L,'gpuArray'),[1,1,N]);
C(2:L-1,1:L-2,:) = C(2:L-1,1:L-2,:)+eye(L-2).*(numer(2:end,:,:)./denom);
C(2:L-1,3:L,:) = C(2:L-1,3:L,:)+eye(L-2).*(numer(1:end-1,:,:)./denom);

% 
% for i=2:L-1
%     C(i,i) = 1;
%     C(i,i-1) = -(x(i+1)-x(i)) / (x(i+1)-x(i-1));
%     C(i,i+1) = -(x(i)-x(i-1)) / (x(i+1)-x(i-1));
% end
% C(1,1) = 1; C(L,L) = 1;
end