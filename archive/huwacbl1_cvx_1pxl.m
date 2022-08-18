function [xtmp,rtmp,btmp] = huwacbl1_cvx_1pxl(A,ytmp,C,cl,L,N,n,varargin)
% display cvx detail or not
verbose = false;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'VERBOSE'
                if strcmp(varargin{i+1},'yes')
                    verbose=true;
                elseif strcmp(varargin{i+1},'no')
                    verbose=false;
                else
                    error('verbose is invalid');
                end
            case 'SOLVER'
                tmp = varargin{i+1};
                cvx_solver tmp
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end
    ytmp_norm = norm(ytmp);
    I = eye(L) * ytmp_norm;
    Atmp = [A I -I];
    cvx_begin
    cvx_quiet(~verbose)
    variable xa(n)
    variable btmp(L)
    minimize( cl'*xa )
    subject to
        Atmp * xa + btmp == ytmp
        xa >= 0
        C*btmp >= 0
    cvx_end
%     Atmp = A(:,2:end); At = A(:,1);
%     cvx_begin quiet
%         variable xt(1)
%         variable xa(N-1)
%         variable btmp(L)
%     minimize( norm(Atmp*xa-At*xt-btmp,1)+0.001*norm(xa,1) )
%     subject to
%         xt >= 0
%         xa >= 0
%         C*btmp >= 0
%     cvx_end
%     xtmp = [xt; xa]; rtmp = Atmp*xa-At*xt-btmp;
    cvx_opts.cvx_cputime = cvx_cputime;
    cvx_opts.cvx_optbnd = cvx_optbnd;
    cvx_opts.cvx_optval = cvx_optval;
    cvx_opts.cvx_precision = cvx_precision;
    cvx_opts.cvx_slvitr = cvx_slvitr;
    cvx_opts.cvx_slvtol = cvx_slvtol;
    cvx_opts.cvx_status = cvx_status;

    xtmp = xa(1:N,:);
    rtmp = [I -I]*xa(N+1:n,:);
end