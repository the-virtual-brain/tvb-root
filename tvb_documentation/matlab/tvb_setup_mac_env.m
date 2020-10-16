function tvb_setup_mac_env(dist_path)
% tvb_setup_mac_env handles setting up a Python environment for Mac OS.
% The required packages are downloaded from the Anaconda website and repos.

tvb_prefix = fileparts(which(mfilename));
tvb_env_path = fullfile(tvb_prefix, 'tvb_env');

if exist(tvb_env_path, 'dir')
    error('TVB Python environment already exists.');
end

fprintf('TVB Python environment will be set up.\n');

installer_path = fullfile(tvb_prefix, 'tvb_python_installer.sh');

if ~exist(installer_path, 'file')
    base_url = 'https://repo.continuum.io/miniconda/';
    mac_url = 'Miniconda-latest-MacOSX-x86_64.sh';
    fprintf('downloading tvb python installer..\n');
    installer_path = websave(installer_path, [base_url mac_url]);
end

cmd_fmt = 'bash %s -b -p %s';
system(sprintf(cmd_fmt, installer_path, tvb_env_path));

% install packages not trivially linked to in distribution
system('tvb_env/bin/conda install -y -q numpy nomkl numba scipy numexpr h5py')

if nargin < 1
    [here, ~, ~] = fileparts(mfilename);
    dist_path = fullfile(here, '..', 'tvb.app', 'Contents', 'Resources', 'lib', 'python3.7');
end

site_pkgs_path = fullfile(dist_path, 'tvb.app/Contents/Resources/lib/python3.7')
site_pkgs = dir(site_pkgs_path);

target_path = 'tvb_env/lib/python3.7';

for i=1:length(site_pkgs)
    cmd_fmt = 'ln -s %s/%s %s/';
    cmd = sprintf(cmd_fmt, site_pkgs_path, site_pkgs(i).name, ...
        target_path);
    system(cmd);
end
