function tvb_setup

[version, executable, isloaded] = pyversion;

if ~isloaded
    
    % in distribution, try to find path relative to this file
    tvb_path = 'C:\Users\mw\Downloads\TVB_Distribution';

    if isempty(tvb_path)
        error('Please set the path to TVB_Distribution folder in tvb_setup.m')
    end

    pyversion(fullfile(tvb_path, 'tvb_data', 'python.exe'))
else
    fprintf('[tvb_setup] using Python %s %s\n', version, executable);
end

py.tvb_matlab.setup()
py.tvb.simulator.common.log_debug(py.False, py.False, '[TVB]')