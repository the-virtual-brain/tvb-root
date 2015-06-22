import numpy
import scipy.io

tvb_data_path = '/home/sophie/dev/tvb_data/tvb_data/'
eeg_chan_path = 'brainstorm/data/TVB-Subject/EEG_channels/channel.mat'
mat = scipy.io.loadmat(tvb_data_path + eeg_chan_path)

name = [l[0] for l in mat['Channel']['Name'][0]]
loc = numpy.array([l[:, 0] for l in mat['Channel']['Loc'][0]])

with open(tvb_data_path + 'sensors/eeg-brainstorm-65.txt', 'w') as fd:
    for n, (x, y, z) in zip(name, loc):
        fd.write('\t'.join(map(str, (n, x, y, z))) + '\n')