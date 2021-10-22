function [] = suwab_toolbox_startup_addpath()
%-------------------------------------------------------------------------%
% % Automatically find the path to toolboxes
fpath_self = mfilename('fullpath');
[dirpath_self,filename] = fileparts(fpath_self);
mtch = regexpi(dirpath_self,'(?<parent_dirpath>.*)/suwab_toolbox[/]{0,1}','names');
toolbox_root_dir = mtch.parent_dirpath;

%-------------------------------------------------------------------------%
% name of the directory of each toolbox
suwab_toolbox_dirname         = 'suwab_toolbox';
%-------------------------------------------------------------------------%
pathCell = strsplit(path, pathsep);

%% base toolbox
% base_toolbox_dir = [toolbox_root_dir '/' base_toolbox_dirname ];
% % joinPath in base toolbox will be used in the following. "base" toolbox
% % need to be loaded first. base/joinPath.m automatically determine the
% % presence of trailing slash, so you do not need to worry it.
% if exist(base_toolbox_dir,'dir')
%     if ~check_path_exist(base_toolbox_dir, pathCell)
%         addpath(base_toolbox_dir);
%     end
% else
%     warning([ ...
%         'base toolbox is not detected. Download from' '\n' ...
%         '   https://github.com/yukiitohand/base/'
%         ]);
% end

%% suwab_toolbox
suwab_toolbox_dir = joinPath(toolbox_root_dir, suwab_toolbox_dirname);
if ~check_path_exist(suwab_toolbox_dir, pathCell)
    addpath( ...
        suwab_toolbox_dir                     , ...
        joinPath(suwab_toolbox_dir,'legacy/') , ...
        joinPath(suwab_toolbox_dir,'dev/')      ...
    );
end

end

%%
function [onPath] = check_path_exist(dirpath, pathCell)
    % pathCell = strsplit(path, pathsep, 'split');
    if ispc || ismac 
      onPath = any(strcmpi(dirpath, pathCell));
    else
      onPath = any(strcmp(dirpath, pathCell));
    end
end