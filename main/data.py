import numpy as np
import h5py


class Data:
    def __init__(self, folder, filename):
        self.write_filename = folder + filename + '.hdf5'
        self.info_name = folder + filename + '_info.txt'

    def create_file(self, distribution, density, field):
        # Open file for writing
        with h5py.File(self.write_filename, 'w') as f:
            # Create datasets, dataset_distribution =
            f.create_dataset('pdf', data=np.array([np.real(distribution)]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1], distribution.shape[2]),
                             dtype='f')
            f.create_dataset('density', data=np.array([np.real(density)]),
                             chunks=True,
                             maxshape=(None, density.shape[0]),
                             dtype='f')
            f.create_dataset('field', data=np.array([np.real(field)]),
                             chunks=True,
                             maxshape=(None, field.shape[0]),
                             dtype='f')

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            pdf = f['pdf'][()]
            den = f['density'][()]
            eng = f['field'][()]
        return pdf, den, eng
