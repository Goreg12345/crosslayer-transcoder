import torch
from torch.utils.data import Dataset
import h5py


class DiscBuffer(Dataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It provides
    random access to the elements stored in an HDF5 file.
    """

    def __init__(self, h5_path, accessor="tensor", **kwargs):
        self.h5_path = h5_path
        self.accessor = accessor
        # Open the HDF5 file in read mode for faster access
        self.h5 = h5py.File(self.h5_path, "r", **kwargs)
        self.buffer = self.h5[self.accessor]

    def __len__(self):
        # Return the total number of items in the dataset
        return self.buffer.shape[0]

    @torch.no_grad()
    def __getitem__(self, idx):
        # Fetch the data item at the specified index
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Efficiently fetch the data from the HDF5 file
        data = torch.tensor(self.buffer[idx], dtype=torch.float32)
        return data

    def close(self):
        # Close the HDF5 file when done
        if self.h5:
            self.h5.close()