% This is a utility script for regenerating TVB demo HTML files.
% It requires pandoc to be installed to generate RST from HTML files.
%
% NB It clears the workspace and closes all figures before starting

clear all
close all

demo_list = dir('tvb_demo_*');
html_results = {};
rst_results = {};
for i=1:length(demo_list)
    demo_name = demo_list(i).name;
    
    % generate HTML for demo
    html_results{i} = publish(demo_name);
    
    % pandoc HTML to RST
    [path, base, ~] = fileparts(html_results{i});
    rst_results{i} = [fullfile(path, base) '.rst'];
    cmd_fmt = 'pandoc %s -o %s';
    cmd = sprintf(cmd_fmt, html_results{i}, rst_results{i})
    system(cmd)
    close all
end

%% Write list of rst files
fd_demos = fopen('../tvb_documentation/demos/Demos_Matlab.rst', 'w');
include_fmt = '.. include:: ../matlab/html/%s.rst\n\n';
for i=1:length(rst_results)
    
    % add rst to list of demos
    rst_fname = rst_results{i};
    [~, base, ~] = fileparts(rst_fname);
    fprintf(fd_demos, include_fmt, base);
    
    % read rst file lines
    lines = {};
    fd = fopen(rst_fname, 'r');
    l = fgetl(fd);
    while ischar(l);
        lines = {lines{:} l};
        l = fgetl(fd);
    end
    fclose(fd);
    
    % rewrite, with ref target
    fd = fopen(rst_fname, 'w');
    fprintf(fd, '.. _%s:\n\n', base);
    for i=1:length(lines)
        fprintf(fd, '%s\n', lines{i});
    end
    fclose(fd);
    
end

fclose(fd_demos);