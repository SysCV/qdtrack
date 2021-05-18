import h5py
import mmcv
import numpy as np
import os
from mmcv import BaseStorageBackend, FileClient


@FileClient.register_backend('hdf5')
class HDF5Backend(BaseStorageBackend):

    def __init__(self, vid_db_path, img_db_path="", **kwargs):
        self.vid_db_path = str(vid_db_path)
        self.img_db_path = str(img_db_path)
        self.vid_client = None
        self.img_client = None

    def get(self, filepath):
        """Get values according to the filepath.
        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
        """
        filepath = str(filepath)
        filefolder, filename = os.path.split(filepath)
        if filefolder == "":
            if self.img_client is None:
                self.img_client = h5py.File(self.img_db_path, 'r')
            value_buf = np.array(self.img_client[filename])
        else:
            if self.vid_client is None:
                self.vid_client = h5py.File(self.vid_db_path, 'r')
            group = self.vid_client[filefolder]
            value_buf = np.array(group[filename])
        return value_buf

    def get_text(self, filepath):
        raise NotImplementedError

