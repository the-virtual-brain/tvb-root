import os.path
import subprocess
from tvb.datatypes.volumes import Volume, VolumeInFile, FreeSurferSubject
from tvb.basic.neotraits.api import HasTraits, Attr

class FreeSurferRun(HasTraits):
    input_T1 = Volume(label="Input T1.")
    def evaluate(self):
        vol: VolumeInFile = VolumeInFile.from_volume(self.input_T1)
        path = vol.file_path
        tmp_dir = '/tmp/tvb-fs'
        cmd = 'freesurfer -i {} -all -s {}'
        proc = subprocess.Popen(
            cmd.format(path, 'tvb').split(),
            env={'SUBJECTS_DIR': tmp_dir},
            stderr=subprocess.STDIN,
            stdin=subprocess.PIPE,
        )
        proc.wait()
        return FreeSurferSubject(
            subject_folder=os.path.join(tmp_dir, 'tvb'))