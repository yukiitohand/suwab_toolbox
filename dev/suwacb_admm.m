function [ XA,XC,res,Ctmp ] = suwacb_admm( x,A,Y,varargin )
% [ XA,XC ] = suwacb( x,A,Y,varargin )
%   sparse unmixing with adaptive concave background
%   Input variables
%       x: (d x 1)-array, wavelengths for spectral channels
%       A: (d x p)-array, each column corresponds to endmembers, d: number
%          of channels, p: number of endmembers
%       Y: (d x N)-array, each column is an observed spectrum.
%   Optional parameters
%       'TOL': scalar, tolerance, (default) 0.048
%       'MAXNUM': scalar, maximal number of endmembers, (default) 4
%       'PRESELECTED_IDX': logical or numerical array
%            indicies for the preselected endmembers, (default) []
%       'PRESELECT_MODE': char/string {"BATCH0","N-SEQUENTIAL"}
%            mode for how to deploy pre-selected endmembers
%            (default) "Normal"
%            "Normal": no preselection.
%            "BATCH0"
%              The pre-selected endmembers are integrated into the library
%              at the iteration 0 and later, considered to be a part of
%              background.
%            "N-SEQUENTIAL"
%              First NPRE iterations are performed only with the pre-selected
%              endmembers.
%       'NPRE': scalar,
%          Only used with "N-SEQUENTIAL" mode. Define the number of
%          iterations with the pre-selected endmembers. If Npre > MAXNUM,
%          then the iteration stops at MAXNUM.
%          (default) 0
%
%   [future implementation]
%       'BLOCKSIZE': scaler, 
%   Output variables
%       XA: (p x N)-array, estimated abundance matrix
%       XC: (d x N)-array, estimated concave curve matrix
%       res: (1 x N)-array, residuals

tol = 0.044;
maxnum = 4;
idx_sel_init = [];
lambda2_c = 0;
mode_preselect = 'Normal';
Npre = 0;

if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs.');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'PRESELECT_MODE'
                mode_preselect = varargin{i+1};
            case 'NPRE'
                Npre = varargin{i+1};
            case 'PRESELECTED_IDX'
                idx_sel_init = varargin{i+1};
            case 'LAMBDA2_C'
                lambda2_c = varargin{i+1};
            case 'TOL'
                tol = varargin{i+1};
                if (tol<=0)
                       error('TOL must be positive.');
                end
            case 'MAXNUM'
                maxnum = varargin{i+1};
                % if (maxnum<=0)
                %        error('MAXNUM must be positive.');
                % end
            otherwise
                error('Unrecognized option: %s',varargin{i});
        end
    end
end

[d,p] = size(A); N = size(Y,2);

if ~isempty(idx_sel_init)
    if islogical(idx_sel_init)
        if length(idx_sel_init) ~= p
            error('The size of PRESELECTED_IDX is not right');
        end
        idx_sel_init = reshape(find(idx_sel_init),1,[]);
    end
end

switch upper(mode_preselect)
    case 'NORMAL'
        idx_sel_init = []; idx_res_init = 1:p;
    case 'BATCH0'
        if ~isempty(idx_sel_init)
            idx_res_init = setdiff(1:p,idx_sel_init);
        else
            idx_sel_init = []; idx_res_init = 1:p;
        end
    case 'N-SEQUENTIAL'
        if ~isempty(idx_sel_init)
            idx_res_init = idx_sel_init;
            idx_sel_init = [];
            if Npre==0
                fprintf('No point in doing %s with Npre=%d...\n',...
                    mode_preselect,Npre);
            end
        else
            idx_sel_init = []; idx_res_init = 1:p;
        end
        idx_res_all = 1:p;
    otherwise
        error('Undefined PreselectMode %s',mode_preselect);
end

XA = zeros(p,N); XC = zeros(d,N); res = zeros(1,N);

for n=1:N %numblocks
    y = Y(:,n);
    % ------------------------------------------------------------------- %
    % first iteration with pre-selected endmembers
    % ------------------------------------------------------------------- %
    % start from pre-selected endmembers
    idx_sel = idx_sel_init; idx_res = idx_res_init;
    Atmp = A(:,idx_sel);
    
    % Perform a constrained non-negative least squares with pre-selected
    % endmembers
    [xA_min,xC_min,Ctmp] =  huwacb_admm2_stflip(Atmp,y,x,...
        'lambda2_c',lambda2_c,'tol',1e-5','maxiter',1000,'verbose','no');
    
    % Obtain residual
    if isempty(xA_min)
        r_min = vnorms(y - Ctmp*xC_min,1,2);
    else
        r_min = vnorms(y - Atmp*xA_min - Ctmp*xC_min,1,2);
    end
    fprintf('Initial Error: %f\n',r_min);

    % ------------------------------------------------------------------- %
    % Main Loop
    % ------------------------------------------------------------------- %
    iter = 0;
    % Aggregate endmembers if the error "r_min" is greater than tolerance
    % or until the maximum number of the additional endmembers is reached.
    while (r_min>tol) && (iter < maxnum)
        iter = iter + 1;
        
        switch upper(mode_preselect)
            case 'N-SEQUENTIAL'
                % After finishing first Npre iterations, preset the
                % unselected endmembers.
                if iter==Npre+1
                    idx_res = setdiff(idx_res_all,idx_sel);
                end
        end
        
        % Find the endmember that minimizes the error in a least square
        % sense.
        r_min=inf; i_r_min = 0;
        for i=idx_res
            % increment an endmember
            Atmp = [A(:,idx_sel) A(:,i)];
            
            % Perform a constrained non-negative least squares with the
            % incremented endmember.
            [xA,xC,Ctmp] =  huwacb_admm2_stflip(Atmp,y,x,...
                'lambda2_c',lambda2_c,'tol',1e-5','maxiter',1000);
            
            % Obtain residual
            r = vnorms(y - Atmp*xA - Ctmp*xC,1,2);
            
            % if the obtained error is smaller than current minimizer,
            % exchange the candidate.
            if r<r_min
                r_min = r; i_r_min = i; xA_min = xA; xC_min = xC;
            end
        end
        fprintf('Iter:%d, Index:%d selected\nError: %f\n',iter,i_r_min,r_min);
        
        % Add the endmember that reduces the error most.
        idx_sel = [idx_sel i_r_min];
        idx_res = setdiff(idx_res,i_r_min);
    end
    
    if ~isempty(xA_min), XA(idx_sel,n) = xA_min; end
    
    XC(:,n) = xC_min; res(n) = r_min;
    
end

end