function [x,z,C,r,d,outs] = aradmm_huwacbl1(A,y,wv,lambda_a)

% demo linear regression with elastic net regularizer
% use the ARADMM solver
% details in Adaptive Relaxed ADMM: Convergence Theory and Practical
% Implementation, CVPR 2017
% @author: Zheng Xu, xuzhustc@gmail.com

%% 
% minimize  lam*|x|_1 + |y-Ax-Cw|_1
% subject to x>=0, w[2:L-1]>=0

% formulated as
%     minimize  |c1.*s|_1
%     subject to s>=c2, Ts=y
% where      _   _
%           |  x  |
%      s =  |  w  |, T =  [ A C I_L],
%           |_ r _|
%                                       _         _ 
%            _          _              |   0_N     |
%           |  lambda_a  |             |  -inf     |
%      c1 = |    0_L     |,  and  c2 = |   0_{L-2} |
%           |_   1_L    _|             |  -inf     |
%                                      |_  0_L    _|
%
% 
% variable augmented
%     minimize  |c1.*t|_1
%     subject to t>=c2, Ts=y, s-t=0
%  
%     minimize  |c1.*t|_1 + I_{t>=c2}(t) + I_{Ts=y}(s)
%     subject to s-t=0

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check validity of input parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mixing matrixsize
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


rho = 0.01;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the bases for continuum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = continuumDictionary(wv);
s_c = vnorms(C,1);
C = bsxfun(@rdivide,C,s_c);
C = C*2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-processing for main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ynorms = vnorms(y,1);
tau=ynorms;
tau1 = 0.2;
if Aisempty
    T = [C tau1*eye(L)];
else
    T = [A C tau1*eye(L)];
end
TTt = T*T';
Tpinvy = T' * (TTt \ y);
PT_ort = eye(N+2*L) - T' * (TTt \ T);
% projection operator
c1 = zeros([N+2*L,Ny]);
c1(1:N,:) = lambda_a.*ones([N,Ny]);
c1(N+L+1:N+L*2,:) = ones([L,1])./tau*tau1;
% c1(N+L+1:N+L*2) = ones([L,1])*tau1;
% c1rho = c1./rho;

c2 = zeros([N+2*L,1]);
c2(N+1) = -inf; c2(N+L) = -inf; c2(N+L+1:N+2*L) = -inf;

obj = @(s,t) sum(vnorms(c1.*t,1,1));

solvh = @(t,d,rho) PT_ort*(t+d./rho)+Tpinvy;
solvg = @(s,d,rho) max(soft_thresh(s-d./rho,c1./rho),c2);
fA = @(x) x;
fAt = @(x) x;
fB = @(x) -x;
fb = 0;




%% initialization
s0 = Tpinvy;
t0 = max(soft_thresh(s0,c1./rho),c2);
d0 = -s0+t0;

%%
opts = [];
opts.tol = sqrt((L*2+N)*Ny)*1e-6; %stop criterion, relative tolerance
opts.maxiter = 120; %max interation
opts.tau = rho; %initial stepsize
%verbose print
%0: no print,
%1: print every iteration
%2: evaluate objective every iteration
%3: more print out for debugging adaptive relaxed ADMM
opts.verbose = 3;
opts = get_default_opts(opts);
opts.adp_flag = 5;
opts.gamma = 1;

opts.obj = obj; %objective function, used when verbose

%%
% ADMM solver
tic;
[sol, outs] = aradmm_core(solvh, solvg, fA, fAt, fB, fb, t0, d0, opts);
outs.runtime  = toc;
% sol = sol.u;
x = sol.v(1:N,:);
z = sol.v(N+1:N+L,:);
r = sol.v(N+L+1:N+L*2,:);
d = sol.l;

end
