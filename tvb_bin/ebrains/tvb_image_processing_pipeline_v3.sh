#!/bin/bash

######## MAIN TVB PIPELINE ORCHESTRATOR SCRIPT
# Generic script to orchestrate containers on
# supercomputers in a secure and reproducible manner.
# Authors: Michael Schirner & Petra Ritter
#          Charité--Universitätsmedizin Berlin, Germany
#          Brain Simulation Section (PI: Petra Ritter)
version="3.0.0" # Sets version variable
# HISTORY:
#
# * Early 2020 - v1.0.0  - Initial Version: clean fMRI, convert to TVB format (developed by Paul Triebkorn)
# * Late 2020  - v1.5.0  - tvb_converter is now orchestrator of the pipeline and supports data privacy
# * July 2021  - v2.0.0  - added DataLad for reproducibility
# * July 2021  - v3.0.0  - revamp: replace complex encryption with something simpler, added GUI support
#
# NOTES:
# ##################################################

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# FIXME: set some arguments manually for now
source_ds="pipeline"
subid="sub-CON04"
outputstore_folder="outputstore"
inputstore_folder="inputstore"
JOBID="testjob"
containername1='tvbpipe-mrtrix3'
containername2='tvbpipe-fmriprep'
containername3='tvbpipe-converter'
container_repoimage1="docker://bids/mrtrix3_connectome:latest"
container_repoimage2="docker://poldracklab/fmriprep:latest"
container_repoimage3="docker://michamischa/tvb-pipeline-converter:1.2"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



# Enable xtrace if the DEBUG environment variable is set
if [[ ${DEBUG-} =~ ^1|yes|true$ ]]; then
    set -o xtrace       # Trace the execution of the script (debug)
fi

# Exit on error
set -o errexit          # Exit on most errors (see the manual)
set -o errtrace         # Make sure any error trap is inherited
set -o nounset          # Disallow expansion of unset variables
set -o pipefail         # Use last non-zero exit code in a pipeline

# array to store secrets
declare -A secrets=()

# generic logging format
log(){
    local msg="$1"
    timeAndDate=`date`
    echo "[$timeAndDate] $msg" >> $logFile
}

# DESC: Handler for unexpected errors
# ARGS: $1 (optional): Exit code (defaults to 1)
# OUTS: None
script_trap_err() {
    log "script_trap_err(): Trapped error. Pipeline will now exit."
    local exit_code=1

    # Disable the error trap handler to prevent potential recursion
    trap - ERR

    # Consider any further errors non-fatal to ensure we run to completion
    set +o errexit
    set +o pipefail

    # Remove tmp directory
#    if [ ! -z ${tmpDir+x} ]; then
#        if [ -d "${tmpDir}" ]; then
#            rm -r "${tmpDir}"
#            log "script_trap_err(): temporal directory removed: ${tmpDir}"
#        fi
#    fi

    # Validate any provided exit code
    if [[ ${1-} =~ ^[0-9]+$ ]]; then
        exit_code="$1"
    fi

    # Exit with failure status
    exit "$exit_code"
}

# DESC: Handler for exiting the script
# ARGS: None
# OUTS: None
script_trap_exit() {
    log "script_trap_exit(): Trapped exit. Pipeline will now exit."
    # Remove tmp directory
#    if [ ! -z ${tmpDir+x} ]; then
#        if [ -d "${tmpDir}" ]; then
#            rm -r "${tmpDir}"
#            log "script_trap_exit(): temporal directory removed: ${tmpDir}"
#        fi
#    fi

    cd "$orig_cwd"
}

# DESC: Exit script with the given message
# ARGS: $1 (required): Message to print on exit
#       $2 (optional): Exit code (defaults to 0)
# OUTS: None
# NOTE: The convention used in this script for exit codes is:
#       0: Normal exit
#       1: Abnormal exit due to external error
#       2: Abnormal exit due to script error
script_exit() {
    log "script_exit(): Pipeline will now exit."
    # Remove tmp directory
#    if [ ! -z ${tmpDir+x} ]; then
#        if [ -d "${tmpDir}" ]; then
#            rm -r "${tmpDir}"
#            log "script_exit(): temporal directory removed: ${tmpDir}"
#        fi
#    fi

    if [[ $# -eq 1 ]]; then
        printf '%s\n' "$1"
        exit 0
    fi

    if [[ ${2-} =~ ^[0-9]+$ ]]; then
        printf '%b\n' "$1"
        # If we've been provided a non-zero exit code run the error trap
        if [[ $2 -ne 0 ]]; then
            script_trap_err "$2"
        else
            exit 0
        fi
    fi

    script_exit 'Missing required argument to script_exit()!' 2
}



# DESC: Generic script initialisation
# ARGS: $@ (optional): Arguments provided to the script
# OUTS: $orig_cwd: The current working directory when the script was run
#       $script_path: The full path to the script
#       $script_dir: The directory path of the script
#       $script_name: The file name of the script
#       $script_params: The original parameters provided to the script
# NOTE: $script_path only contains the path that was used to call the script
#       and will not resolve any symlinks which may be present in the path.
#       You can use a tool like realpath to obtain the "true" path. The same
#       caveat applies to both the $script_dir and $script_name variables.
# shellcheck disable=SC2034
script_init() {
    # Useful paths
    readonly orig_cwd="$PWD"
    readonly script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
    readonly script_name=`basename "$0"`
    readonly script_path="$script_dir/$script_name"
    readonly script_params="$*"
    echo "script_init(): TVB Pipeline initialized"
}


# Print usage manual
script_usage() {
    cat << EOF
The TVB Processing Pipeline has five usage modes:
Mode 0: Install DataLad via Miniconda.
Mode 1: Pull containers and create DataLad container datasets.
Mode 2: Create DataLad analysis dataset with container and inputs as subdatasets.
Mode 3: Create and submit main SLURM job.
Mode 4: Run workflow on compute node.

Usage:
    ./$script_name -m <mode> -p <path/to/workingdir>

    -m <mode> | Select usage mode
    -p        | Path to working directory
    -h        | Display this help

Example:

./tvb_image_processing_pipeline_v3.sh -m 1 -p /users/johndoe/myworkfolder

EOF
}



# Parameter parser
parse_params() {
    mode=-1
    path_provided=false

    # specify options and whether there are arguments expected (:)
    options='m:p:h'

    while getopts $options option
    do
        case "$option" in
            m  ) mode=$OPTARG;;
            p  ) path_provided=true; working_dir=$OPTARG;;
            h  ) script_usage; exit 0;;
            \? ) echo "Unknown option: -$OPTARG" >&2; exit 1;;
            :  ) echo "Missing option argument for -$OPTARG" >&2; exit 1;;
            *  ) echo "Unimplemented option: -$OPTARG" >&2; exit 1;;
        esac
    done

    # No arguments were provided. Exit.
    if [ $# -lt 1 ]; then
        echo "Error: No arguments provided. Aborting." >&2
        script_usage
        exit 1
    fi

    # check if mode number is valid (must be from 1-4)
    if ((mode > 5 || mode < 0)); then
        echo "Error: Invalid usage mode "${mode}". Mode must be a number between 1 and 4." >&2
        script_usage
        exit 1
    fi

    # Make sure that working directory was specified.
    if ! $path_provided; then
        echo "Error: No path to working directory provided. Aborting." >&2
        script_usage
        exit 1
    fi

    # check if working directory does not exist and create if not
    if [ ! -d "$working_dir" ]; then
        mkdir "$working_dir"
    fi

    # check again if working directory exists and abort if not
    if [ ! -d "$working_dir" ]; then
        echo "Error: Working directory does not exist and cannot be created." >&2
        exit 1
    fi

    # work in "$working_dir"
    cd "$working_dir"

    # Input argument parsing successful, start logging.
    log "******THE VIRTUAL BRAIN PROCESSING PIPELINE******"
    log "parse_params(): Starting mode ${mode} in ${working_dir}."
}


# Install DataLad via Miniconda and pip and set up Git identity
install_datalad() {
    echo "install_datalad(): Installing DataLad via Miniconda." >&2
    log "install_datalad(): Installing DataLad via Miniconda."

    cd "$working_dir"

    # FIXME: modules too specific for Piz Daint
    module purge
    module load daint-mc # supercomputer-specific module
    module load cray-python/3.8.5.0 # load Python environment

    # The easiest way to install DataLad with all
    # dependencies on a supercomputer without root
    # permissions is by using conda
    # FIXME: Version may change
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    # acknowledge license, initialize Miniconda3, close and
    # re-open shell
    conda install -c conda-forge datalad

    # DataLad extensions are required
    pip install datalad-neuroimaging datalad-container

    # Set up Git Identity
    git config --global --add user.name "TVBPIPELINEservice"
    git config --global --add user.email tvb_pipeline@ebrains.eu

    # delete installer
    rm Miniconda3-latest-Linux-x86_64.sh

    echo "install_datalad(): Done." >&2
    log "install_datalad(): Done."
    exit 0
}


# Generate RSA keys and stores them in two files. The private key file is protected by a password and encrypted with AES128-CBC and 'scrypt' to thwart dictionary attacks.
generate_keys() {
    log "generate_keys(): generating keys..."
    secrets[$1]="${RANDOM}_${RANDOM}_${RANDOM}_${RANDOM}" # This password must remain secret!

    # Load modules and run tvb_converter container to generate keys
    module purge
    module load daint-mc
    module load sarus
#    srun -C mc --account=ich012 sarus run --mount=type=bind,source="$working_dir"/keys,destination=/keys michamischa/tvb-pipeline-converter:1.0 /tvb_converter.sh -k $passphrase -o /keys
#    sarus run --mount=type=bind,source="$working_dir"/keys,destination=/keys michamischa/tvb-pipeline-converter:1.0 /tvb_converter.sh -k $passphrase -o /keys

    sarus run --entrypoint "" --mount=type=bind,source="$working_dir"/keys,destination=/keys michamischa/tvb-pipeline-converter:1.2 python generateKeys.py ${secrets[$1]} /keys

#    ls -ltr "$working_dir"/keys
    mv "$working_dir"/keys/private_key.bin "$working_dir"/keys/private_key_${1}.bin
    mv "$working_dir"/keys/public_key.pem "$working_dir"/keys/public_key_${1}.pem

    log "generate_keys(): $1 keys generated."
}

slurm_header() {
    printf "#!/bin/bash -l\n#SBATCH --account=ich012\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --output=job-${1}.out\n#SBATCH --error=job-${1}.err\n#SBATCH --time=${2}\n#SBATCH --constraint=mc\nmodule load daint-mc\nmodule load sarus\n"
}

# Pull the three pipeline containers
pull_containers() {
    echo "pull_containers(): Generating batch files to pull containers." >&2
    log "pull_containers(): Generating batch files to pull containers."

    # specify header of slurm batch job files
    head1=$(slurm_header "SC" "05:00:00")
    head2=$(slurm_header "FC" "05:00:00")
    head3=$(slurm_header "Co" "05:00:00")

    # create slurm batch job files
    cat <<EOF > pull_c1.sh
${head1}
#srun sarus pull thevirtualbrain/tvb-pipeline-sc:1.0
srun sarus pull bids/mrtrix3_connectome
EOF

    cat <<EOF > pull_c2.sh
${head2}
srun sarus pull nipreps/fmriprep
#srun sarus pull thevirtualbrain/tvb-pipeline-fmriprep:1.0
EOF

    cat <<EOF > pull_c3.sh
${head3}
srun sarus pull michamischa/tvb-pipeline-converter:1.2
EOF


    echo "pull_containers(): Slurm batch files generated. Submitting jobs..." >&2
    log "pull_containers(): Slurm batch files generated. Submitting jobs..."

    # submit SLURM job files
    sbatch pull_c1.sh
    sbatch pull_c2.sh
    sbatch pull_c3.sh

    echo "pull_containers(): Jobs submitted. Removing batch files." >&2
    log "pull_containers(): Jobs submitted. Removing batch files."

    # delete SLURM job files
    rm pull_c1.sh
    rm pull_c2.sh
    rm pull_c3.sh

    exit 0
}

slurm_header_singularity() {
    printf "#!/bin/bash -l\n#SBATCH --account=ich012\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --output=job-${1}.out\n#SBATCH --error=job-${1}.err\n#SBATCH --time=${2}\n#SBATCH --constraint=mc\nmodule load daint-mc\nmodule load singularity\n"
}

# Pull the three pipeline containers in a DataLad manner
# CAUTION: pulling multiple containers at the same time with
# different SLURM jobs sometimnes fails.
# Therefore, all three are successively pulled in one job.
#
# CAUTION: alternative pull/build may be necessary e.g.
# singularity build /my_images/fmriprep-v1.3.2.simg docker://poldracklab/fmriprep:latest
#
# NOTE: Pulling fails with errors when containers are pulled in parallel.
# Therefore: only sequential pulling within the same job.
pull_containers_DataLad() {
    echo "pull_containers_DataLad(): Generating batch file to pull containers with DataLad." >&2
    log "pull_containers_DataLad(): Generating batch file to pull containers with DataLad."

    # otherwise built may fail due to missing space in e.g. /tmp
    mkdir -p "$containerstore"/tmp-singularity
    mkdir -p "$containerstore"/tmp-singularity-cache
    export SINGULARITY_TMPDIR="$containerstore"/tmp-singularity
    export SINGULARITY_CACHEDIR="$containerstore"/tmp-singularity-cache

    # specify header of slurm batch job files
    head1=$(slurm_header_singularity "coPull" "05:00:00")

    # create slurm batch job files
    cat <<EOF > pull_co.sh
${head1}
export SINGULARITY_CACHEDIR="$containerstore"/tmp-singularity-cache
export SINGULARITY_TMPDIR="$containerstore"/tmp-singularity


# CAUTION: alternative pull/build may be necessary e.g.
# singularity build /my_images/fmriprep-v1.3.2.simg docker://poldracklab/fmriprep:latest

# container 1
cd "$containerstore"
datalad create "${containername1}"
cd "${containername1}"
datalad containers-add "${containername1}" --url "${container_repoimage1}" \
  --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}'

# container 2
cd "$containerstore"
datalad create "${containername2}"
cd "${containername2}"
datalad containers-add "${containername2}" --url "${container_repoimage2}" \
  --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}'

# container 3
cd "$containerstore"
datalad create "${containername3}"
cd "${containername3}"
datalad containers-add "${containername3}" --url "${container_repoimage3}" \
  --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}'
wait
EOF


    echo "pull_containers_DataLad(): Slurm batch file generated. Submitting job..." >&2
    log "pull_containers_DataLad(): Slurm batch file generated. Submitting job..."

    # submit SLURM job files
    sbatch pull_co.sh

    echo "pull_containers_DataLad(): Job submitted. Removing batch file." >&2
    log "pull_containers_DataLad(): Job submitted. Removing batch file."

    # delete SLURM job files
    rm pull_co.sh

    exit 0
}

# Create main job batch scripts
submit_main_job() {
    echo "submit_main_job(): Generating batch files for main job." >&2
    log "submit_main_job(): Generating batch files for main job."

    # specify header of slurm batch job files
    head1=$(slurm_header "main" "23:55:00")

    # create slurm batch job files
    cat <<EOF > main_job.sh
${head1}
chmod 755 ${script_path}
srun ${script_path} -m 4 -p "$working_dir"
EOF

    sbatch main_job.sh

    echo "submit_main_job(): Jobs submitted. Removing batch files." >&2
    log "submit_main_job(): Jobs submitted. Removing batch files."

    rm main_job.sh

    echo "submit_main_job(): Done." >&2
    log "submit_main_job(): Done."
}


# Create Sandbox (currently unused)
# -----------------------------------
# Here, we spawn a Bash process that behaves exactly as outside the sandbox but
# additionally mounts a sandboxed temp directory.
# This directory will contain the unencrypted personal data.
# The temp directory contains three random numbers and the process ID
# in the name. This directory is removed automatically at exit.
# -----------------------------------
start_sandbox() {
    tmpDir="${output_dir}/tmp.$RANDOM.$RANDOM.$RANDOM.$$"
    (umask 077 && mkdir "${tmpDir}") || {
        echo "start_sandbox(): Could not generate temporal directory. Aborting." >&2
        log "start_sandbox(): Could not generate temporal directory. Aborting."
        exit 1
    }
    log "start_sandbox(): temporal directory created: ${tmpDir}"
    log "start_sandbox(): Spawning sandbox and re-starting pipeline."
    echo "start_sandbox(): Spawning sandbox: ${tmpDir}"
    bwrap --die-with-parent --dev-bind / / --tmpfs ${tmpDir} ${script_path} -m 5 -i ${input_dir} -o ${output_dir} -t ${tmpDir}
    echo "start_sandbox(): Sandbox returned. Removing tmp folder. Stopping."
    rm -rf ${tmpDir} # remove tmp dir (also in trap functions above)
    exit 0
}

start_sandbox_job() {
    tmpDir="${output_dir}/tmp.$RANDOM.$RANDOM.$RANDOM.$$"
    (umask 077 && mkdir "${tmpDir}") || {
        echo "start_sandbox_job(): Could not generate temporal directory. Aborting." >&2
        log "start_sandbox_job(): Could not generate temporal directory. Aborting."
        exit 1
    }
    log "start_sandbox_job(): Creating main pipeline job. (${tmpDir})"
    echo "start_sandbox_job(): Creating main pipeline job. (${tmpDir})"

    head=$(slurm_header "main" "00:10:00")
    cat <<EOF > main_job.sh
${head}
bwrap --die-with-parent --dev-bind / / --tmpfs ${tmpDir} ${script_path} -m ${mode} -i ${input_dir} -o ${output_dir} -s ${tmpDir}
EOF

    sbatch main_job.sh
    echo "start_sandbox_job(): Main pipeline job submitted. Stopping."
    log "start_sandbox_job(): Main pipeline job submitted. Stopping."
    exit 0
}



# This function lets the login-node daemon wait until
# the key from the compute node appears in the keys folder
# After 12 hours without sync the script exits.
sync_with_compute_node() {
    log "sync_with_compute_node(): Waiting for compute node..."
    echo "sync_with_compute_node(): Waiting for compute node..."
    SECONDS=0

    rm -f "$working_dir"/keys/public_key_compute.pem
    rm -f "$working_dir"/keys/encrypted_pwd_computenode.bin
    while ! test -f "$working_dir/keys/public_key_compute.pem"; do
        sleep 5
        if (($SECONDS > 43200)); then
            log "sync_with_compute_node(): Synchronization failed after ${SECONDS} seconds. Stopping now."
            exit 1
        fi
    done
    log "sync_with_compute_node(): ...received public key."
    echo "sync_with_compute_node(): ...received public key."
}


# Here the login node daemon encrypts the password for the private key
# for decrypting the input data on the compute node using the public
# key that was just produced on the compute node.
encrypt_password() {
    sarus run --entrypoint "" --mount=type=bind,source="$working_dir"/keys,destination=/keys michamischa/tvb-pipeline-converter:1.2 python encrypt_secret.py ${secrets[$1]} /keys /keys/public_key_compute.pem

    log "encrypt_password(): Password for compute node encrypted."
    echo "encrypt_password(): Password for compute node encrypted."
}

# This function decrypts the freshly produced "$working_dir"/keys/private_key_input_pwd.bin
# which contains the password for decrypting the private key (private_key_input.bin)
# for decrypting the input data (encrypted_input_data.aes).
decrypt_data() {
    log "decrypt_data(): Waiting for login node to encrypt password..."
    echo "decrypt_data(): Waiting for login node to encrypt password..."
    SECONDS=0

    while ! test -f "$working_dir/keys/private_key_input_pwd.bin"; do
        sleep 5
        if (($SECONDS > 43200)); then
            log "decrypt_data(): Synchronization failed after ${SECONDS} seconds. Stopping now."
            exit 1
        fi
    done
    log "decrypt_data(): ...received encrypted password."
    echo "decrypt_data(): ...received encrypted password."

    # The encrypted password for the input data private key arrived, now let's first decrypt this password,
    # then the private key file, and lastly the data.

    # $tmpDir is automatically deleted if the script exits (see e.g. script_trap_err())
    #tmpDir=$(mktemp -d -p "$working_dir")  || exit 1
    tmpDir="$working_dir"/input-data
    mkdir -p "$tmpDir"
    module purge
    module load daint-mc
    module load sarus
    errormessage=$( sarus run --entrypoint "" --mount=type=bind,source="$working_dir",destination=/input --mount=type=bind,source=${tmpDir},destination=/data michamischa/tvb-pipeline-converter:1.2 python decrypt_data.py /input/keys/private_key_compute.bin ${secrets["compute"]} /input/keys/private_key_input_pwd.bin /input/keys/private_key_input.bin /input/encrypted_password.bin /input/input_data.zip.aes 2>&1 )


    echo $errormessage
    log $errormessage
    log "decrypt_data(): Data decrypted."
    echo "decrypt_data(): Data decrypted."

    cd ${tmpDir}
    unzip data.zip

    log "decrypt_data(): Data unzipped."
    echo "decrypt_data(): Data unzipped."
}




# Create the DataLad superdataset
# incl. containers and input data
create_analysis_dataset_DataLad() {
    log "create_analysis_dataset_DataLad(): Started..."
    echo "create_analysis_dataset_DataLad(): Started..."

    # Load modules, set environment
    module purge
    module load daint-mc
    module load singularity
    export SINGULARITY_CACHEDIR="$containerstore"/tmp-singularity-cache
    export SINGULARITY_TMPDIR="$containerstore"/tmp-singularity

    #vvvvvvvvvvvvvvvv
    #DataLad workflow
    #^^^^^^^^^^^^^^^^

    # Step 1: Turn input into DataLad dataset
    cd "$working_dir"/input-data
    datalad create -f . # create dataset in existing directory
    datalad save . -m "Import all data" # save its contents

    # Step 2: Create an empty analysis dataset where we put all components of the analysis
    cd "$working_dir"
    datalad create -c yoda $source_ds
    cd $source_ds

    # Step 3: Clone the container datasets as subdatasets
    # and register them in the analysis dataset
    # Previously:    --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}'
    containerstore1="$containerstore"/$containername1
    datalad clone -d . ${containerstore1} code/${containername1}
    datalad containers-add \
      --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}' \
      -i "$working_dir"/${source_ds}/code/${containername1}/.datalad/environments/${containername1}/image \
      $containername1

    containerstore2="$containerstore"/$containername2
    datalad clone -d . ${containerstore2} code/${containername2}
    datalad containers-add \
      --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}' \
      -i "$working_dir"/${source_ds}/code/${containername2}/.datalad/environments/${containername2}/image \
      $containername2

    containerstore3="$containerstore"/$containername3
    datalad clone -d . ${containerstore3} code/${containername3}
    datalad containers-add \
      --call-fmt 'singularity run -B {{pwd}} --cleanenv {img} {cmd}' \
      -i "$working_dir"/${source_ds}/code/${containername3}/.datalad/environments/${containername3}/image \
      $containername3

    # amend the previous commit with a nicer commit message
    git commit --amend -m 'Pipeline containers added'


    # Step 4: Import custom code
    cp ~/license.txt code/license.txt
    datalad save -m "Add Freesurfer license file"


    # Step 5: Create input / output siblings
    # These two siblings of the analsis dataset are used to install
    # from and push to two different locations to avoid
    # concurrency and throughput problems
    output_store="ria+file://${working_dir}/${outputstore_folder}"
    input_store="ria+file://${working_dir}/${inputstore_folder}"

    datalad create-sibling-ria -s output "${output_store}"
    datalad create-sibling-ria -s input --storage-sibling off "${input_store}"


    # Step 6: Clone input dataset
    datalad clone -d . "$working_dir"/input-data inputs/data

    # amend the previous commit with a nicer commit message
    git commit --amend -m 'Register input data dataset as a subdataset'


    # Step 7: Save dataset
    datalad save -m "Analysis data set populated."


    # Step 8: cleanup - we have generated the job definitions, no
    # need to keep massive input data files around.
    # (makes Git operations slow, wastes storage space)
    datalad uninstall -r --nocheck inputs/data


    # Step 9: push the fully configured dataset to siblings
    # for initial cloning of inputs and later pushing of results.
    datalad push --to input
    datalad push --to output


    log "create_analysis_dataset_DataLad(): Pipeline setup finished."
    echo "create_analysis_dataset_DataLad(): Pipeline setup finished."
}




# run the pipeline
run_workflow_DataLad() {
    log "run_workflow_DataLad(): Started..."
    echo "run_workflow_DataLad(): Started..."


    #vvvvvvvvvvvvvvvv
    #DataLad workflow
    #^^^^^^^^^^^^^^^^

    # define DSLOCKFILE, DATALAD & GIT ENV for participant_job
    export DSLOCKFILE="$working_dir"/.SLURM_datalad_lock \
    GIT_AUTHOR_NAME=$(git config user.name) \
    GIT_AUTHOR_EMAIL=$(git config user.email) \
    #JOBID=${SLURM_JOB_ID}

    # Step 1: set variables
    cd "$working_dir"/$source_ds
    input_store="ria+file://${working_dir}/${inputstore_folder}"
    dssource="${input_store}#$(datalad -f '{infos[dataset][id]}' wtf -S dataset)"
    pushgitremote="$(git remote get-url --push output)"

    # Step 2: use job-specific temporary folder
    tmpDir="$working_dir"/${JOBID}
    mkdir "$tmpDir"
    cd "$tmpDir"

    # Step 3: get the analysis dataset, which includes the inputs as well
    # IMPORTANT: do not clone from the lcoation that we want to push the
    # results to, in order to avoid too many jobs blocking access to
    # the same location and creating a throughput bottleneck
    datalad clone "${dssource}" ds
    cd ds


    # Step 4: to avoid accumulating temporary git-annex availability information
    # and to avoid a syncronization bottleneck by having to consolidate the
    # git-annex branch across jobs, we will only push the main tracking branch
    # back to the output store (plus the actual file content). Final availability
    # information can be establish via an eventual `git-annex fsck -f joc-storage`.
    # this remote is never fetched, it accumulates a larger number of branches
    # and we want to avoid progressive slowdown. Instead we only ever push
    # a unique branch per each job (subject AND process specific name)
    git remote add outputstore "$pushgitremote"

    # all results of this job will be put into a dedicated branch
    git checkout -b "job-$JOBID"


    # Step 5: obtain input subject manually outside the recorded call, because
    # on a potential re-run we want to be able to do fine-grained recomputing
    # of individual outputs. The recorded calls will have specific paths that
    # will enable recomputation outside the scope of the original setup.
    datalad get -n "inputs/data/${subid}"


    # Step 6: run the containers
    module purge
    module load daint-mc
    module load singularity

    # Weird bug: the former value of $SINGULARITY_BIND was:
    # > echo $SINGULARITY_BIND
    # /opt/cray,/var/opt/cray,/usr/lib64,/lib64,/opt/gcc/9.3.0,/etc/opt/cray/wlm_detect,/var/lib/hugetlbfs
    # but then when the container shall be run singularity complains that "/var/lib/hugetlbfs" is not
    # existent. So we drop it.
    SINGULARITY_BIND="/opt/cray,/var/opt/cray,/usr/lib64,/lib64,/opt/gcc/9.3.0,/etc/opt/cray/wlm_detect"


    mkdir -p .git/tmp/wdir # create workdir for fmriprep
    export SINGULARITY_CACHEDIR="$containerstore"/tmp-singularity-cache
    export SINGULARITY_TMPDIR="$containerstore"/tmp-singularity
    datalad containers-run \
      -m "Compute ${subid}" \
      -n "${containername2}" \
      --explicit \
      -o fmriprep/${subid} \
      -i inputs/data/${subid}/ses-preop/ \
      -i code/license.txt \
      "inputs/data/${subid} . participant --participant-label $subid \
        --anat-only -w .git/tmp/wdir --fs-no-reconall --skip-bids-validation \
        --fs-license-file code/license.txt"


    # Step 7: push resulting file content first. Does not need a lock, no interaction with Git
    datalad push --to output-storage
    # and the output branch next - needs a lock to prevent concurrency issues
    flock --verbose $DSLOCKFILE git push outputstore


    log "run_workflow_DataLad(): Pipeline finished."
    echo "run_workflow_DataLad(): Pipeline finished."
}



# This function encrypts the results with the public key from the data controllers computer after the
# processing of MRI data was finished.
encrypt_results() {
    log "encrypt_results(): Started..."
    echo "encrypt_results(): Started..."

    # compress results into archive
    tar -zcvf ${tmpDir}/results.tar.gz ${tmpDir}/output

    errormessage=$( sarus run --entrypoint "" --mount=type=bind,source="$working_dir"/keys,destination=/key --mount=type=bind,source=${tmpDir},destination=/input --mount=type=bind,source=${output_dir},destination=/output michamischa/tvb-pipeline-converter:1.2 python encrypt_results.py /keys/public_key_results.pem /input/results.tar.gz /output/results.tar.gz.aes /output/results_password.bin 2>&1 )

    echo $errormessage
    log $errormessage

    log "encrypt_results(): Results encrypted."
    echo "encrypt_results(): Results encrypted."
}

# This function deletes all unencrypted outputs
delete_unencrypted_output() {

    # remove the entire temporary folder
    rm -r ${tmpDir}

    log "delete_unencrypted_output(): Unencrypted results deleted."
    echo "delete_unencrypted_output(): Unencrypted results deleted."
}

# DESC: Main control flow
# ARGS: $@ (optional): Arguments provided to the script
# OUTS: None
main() {
    trap script_trap_err ERR
    trap script_trap_exit EXIT

    script_init "$@"
    readonly logFile="${script_dir}/${script_name}.log"
    parse_params "$@"
    cd "$working_dir"
    containerstore="$working_dir"/containers

    # install DataLad
    if ((mode == 0)); then
        install_datalad
        exit 0
    fi

    # pull containers
    if ((mode == 1)); then
        pull_containers_DataLad
        exit 0
    fi

    # create DataLad analysis dataset
    # with containers and input data
    if ((mode == 2)); then
        create_analysis_dataset_DataLad
        exit 0
    fi

    # create main compute job
    if ((mode == 3)); then
        submit_main_job
        exit 0
    fi

    # run workflow on compute node
    if ((mode == 4)); then
        run_workflow_DataLad
        exit 0
    fi

    # generate encryption keys
    if ((mode == 5)); then
        generate_keys "pipeline_keys"
        exit 0
    fi
}

# Start script
main "$@"
exit 0