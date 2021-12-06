import json
import sys 

json_file = sys.argv[1]
call_arg = sys.argv[2]
#json_file = "pipeline_configurations.json"
#call_arg = "fmriprep"
#call_arg = 'estimated_time'
#call_arg = "container"
#call_arg = "participant_label"


# Dict with keys to parse
keys = {}
keys['container'] = ['mrtrix', 'fmriprep', 'tvbconverter']
keys['job'] = ['nr_of_cpus', 'estimated_time']
keys['mrtrix'] = {}
#keys['mrtrix']['preproc'] = ['participant_label', 't1w_preproc']
#keys['mrtrix']['participant'] = ['participant_label', 'session_label', 'parcellation', 'streamlines', 'template_reg', 't1w_preproc']
#keys['mrtrix']['group'] = ['participant_label', 'session_label']
keys['mrtrix']['preproc'] = ['t1w_preproc']
keys['mrtrix']['participant'] = ['parcellation', 'streamlines', 'template_reg', 't1w_preproc']
keys['mrtrix']['group'] = []
keys['mrtrix']['optional'] = ['output_verbosity', 'debug', 'n_cpus', 'skip-bids-validator', 'version']
keys['fmriprep'] = {}
keys['fmriprep']['participant'] = ['participant-label', 'skip_bids_validation', 'anat-only', 'fs-no-reconall']

# Dicts with generic suffixes
suffixes = {}
suffixes['fmriprep'] = "--fs-license-file code/license.txt -w .git/tmp/wdir"


# Functions to add/check keys
def add_arg(input_dict, key, output_list, prefix):
    if key in input_dict:
        output_list.append(prefix + key)
        #print(key, input_dict[key], output_list)
        if input_dict[key] == True:
            return
        if input_dict[key] != False:
            output_list.append(input_dict[key])
        
        
# read json file
with open(json_file) as f:
    data = json.load(f)
    

# Build command lines
if call_arg in ['participant_label', 'task-label', 'session_label', 'estimated_time', 'nr_of_cpus']:
    command_line = [data[call_arg]]
elif call_arg in ['container']:
    command_line = []
    for key in keys[call_arg]:
        add_arg(data, key, command_line, "")
elif call_arg in ['mrtrix', 'fmriprep']:
    analysis_level = data[call_arg + "_parameters"]["analysis_level"]
    command_line = [analysis_level]
    for key in keys[call_arg][analysis_level]:
        add_arg(data, key, command_line, "--")
        add_arg(data[call_arg + "_parameters"]["analysis_level_config"], key, command_line, "--")
    if "optional" in keys[call_arg]:
        for key in keys[call_arg]["optional"]:
            add_arg(data[call_arg + "_parameters"], key, command_line, "--")
    if call_arg in suffixes:
        command_line.append(suffixes[call_arg])
#elif call_arg in ['parcellation']:
#    command_line = [data['mrtrix_parameters']['analysis_level_config'][call_arg]]
    
#print(" ".join(map(str, command_line)))

sys.stdout.write(" ".join(map(str, command_line)))
sys.stdout.flush()
sys.exit(0)

