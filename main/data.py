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
            f.create_dataset('pdf', data=np.array([distribution]),
                             chunks=True,
                             maxshape=(None, distribution.shape[0], distribution.shape[1], distribution.shape[2]),
                             dtype='f')
            f.create_dataset('density', data=np.array([density]),
                             chunks=True,
                             maxshape=(None, density.shape[0]),
                             dtype='f')
            f.create_dataset('field', data=np.array([field]),
                             chunks=True,
                             maxshape=(None, field.shape[0]),
                             dtype='f')
            f.create_dataset('time', data=[0.0], chunks=True, maxshape=(None,))

    def save_data(self, distribution, density, field, time):
        # Open for appending
        with h5py.File(self.write_filename, 'a') as f:
            # Add new time line
            f['pdf'].resize((f['pdf'].shape[0] + 1), axis=0)
            f['density'].resize((f['density'].shape[0] + 1), axis=0)
            f['field'].resize((f['field'].shape[0] + 1), axis=0)
            f['time'].resize((f['time'].shape[0] + 1), axis=0)
            # Save data
            f['pdf'][-1] = distribution
            f['density'][-1] = density
            f['field'][-1] = field
            f['time'][-1] = time

    def read_file(self):
        # Open for reading
        with h5py.File(self.write_filename, 'r') as f:
            time = f['time'][()]
            pdf = f['pdf'][()]
            den = f['density'][()]
            eng = f['field'][()]
        return time, pdf, den, eng
