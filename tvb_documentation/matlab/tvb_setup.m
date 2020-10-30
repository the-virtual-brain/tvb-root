function tvb_setup(python_exe)
% Setup TVB for use. If no argument is given, the distribution Python
% is used (or an env is created on Mac). Otherwise, use the given
% Python executable, and assume the TVB package is on that Python's path.

[version, executable, isloaded] = pyversion;

if ~isloaded
    
    [here, ~, ~] = fileparts(mfilename);
    dist_python = fullfile(here, '..', 'tvb_data', 'python');
    
    if nargin == 0
        switch (computer)
            case 'MACI64'
                try
                    tvb_setup_mac_env
                catch
                end
                pyversion tvb_env/bin/python

            case 'GLNXA64'
                pyversion(dist_python);

            case 'PCWIN'
                fprintf('32-bit Windows unsupported, please upgrade ot 64-bit.\n');

            case 'PCWIN64'
                pyversion([dist_python '.exe']);

        end
        
    else
        pyversion(python_exe)
    end

else
    fprintf('[tvb_setup] using Python %s %s\n', version, executable);
end
%%

append(py.sys.path, fileparts(mfilename('fullpath')))

py.tvb_matlab.setup()
py.tvb.simulator.common.log_debug(py.False, py.False, '[TVB]')
