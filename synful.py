"""Access to the Synful MongoDB and SQL databases."""
from funlib.persistence import open_ds
from funlib.geometry import Roi, Coordinate
import logging 
import numpy as np
from PIL import Image
import psycopg2
import sqlite3
import torch
from torch.utils.data import Dataset



logger = logging.getLogger(__name__)


class SynfulSQLiteDB:
    """
    Access to the Synful SQLite database.
    """
    destination_path = "" # TODO 
    table = "synlinks"
    assumed_length = 244358226

    def __init__(self, use_assumed_length=True, roi=None) -> None:
        self._connection = None
        self._cursor = None
        self._count = None
        if use_assumed_length:
            logger.warning("Using assumed length of the table, which may be inaccurate.")
            self._count = self.assumed_length
        self.modifier = ""
        if roi: 
            self.modifier = "WHERE "

    @property
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.destination_path)
        return self._connection

    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.connection.cursor()
        return self._cursor

    def __len__(self):
        """NOTE: This assumes that the table has not changed."""
        if self._count is None:
            command = f"SELECT COUNT(*) FROM {self.table}"
            self.cursor.execute(command)
            self._count = self.cursor.fetchone()[0]
        return self._count

    def __getitem__(self, item) -> tuple:
        """
        Get pre-synaptic coordinates of the nth synapse.

        Returns
        -------
        tuple
            (x, y, z) coordinates of the pre-synaptic partner.
        """
        command = f"SELECT pre_x, pre_y, pre_z FROM {self.table} WHERE rowid = {item + 1}"
        self.cursor.execute(command)
        try:
            x, y, z = self.cursor.fetchone()
        except TypeError: # No results
            raise IndexError(f"Index {item} out of range.")
        return (x, y, z)

    def __iter__(self):
        """Iterate over all synapses."""
        command = f"SELECT pre_x, pre_y, pre_z FROM {self.table}"
        self.cursor.execute(command)
        for x, y, z in self.cursor:
            yield (x, y, z)


class SynfulPostGreDB:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        database: str = "synful_fafb",
        write_access: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        host: str
            Hostname of the database server.
            Default: "localhost"
        port: int
            Port of the database server.
            Default: 5432
        user: str
            Username for the database.
            Default: "postgres"
        database: str
            Name of the database to access.
            Default: "synful_synapses"
        write_access: bool
            Whether to allow writing to the database.
            Default: False
        """
        self.host = host
        self.port = port
        self.user = user
        self.database = database
        self._connection = None
        self._cursor = None
        self._length = None
        self.synapse_command = "INSERT INTO synapses VALUES"
        self.neuron_command = "INSERT INTO neurons VALUES"
        self.connection_command = "INSERT INTO connections VALUES"
        self.prediction_command = "INSERT INTO nt_predictions VALUES"
        self.score_command = "INSERT INTO scores VALUES"
        self.__write_access = write_access
        if self.__write_access:
            logger.warning("This instance can modify the database. This may be dangerous.")

    @property
    def connection(self):
        if self._connection is None:
            self._connection = psycopg2.connect(
                host=self.host, port=self.port, user=self.user, database=self.database
            )
        return self._connection

    @property
    def cursor(self):
        if self._cursor is None:
            self._cursor = self.connection.cursor()
        return self._cursor

    def __getitem__(self, item) -> dict:
        """Returns the x, y, z coordinates of the nth synapse.

        These will be the pre-synaptic coordinates.
        """
        # TODO add filtering by score and cleft score
        command = f"SELECT pre_x, pre_y, pre_z, id FROM synapses WHERE id = {item}"
        self.cursor.execute(command)
        x, y, z, _ = self.cursor.fetchone()
        # return {"x": x, "y": y, "z": z}
        return x, y, z

    def __len__(self):
        if self._length is None:
            command = "SELECT COUNT(*) FROM synapses"
            self.cursor.execute(command)
            self._length = self.cursor.fetchone()[0]
        return self._length


class SynfulFAFBDataset(SynfulPostGreDB, Dataset):
    container = "/nrs/saalfeld/FAFB00/v14_align_tps_20170818_dmg.n5"
    ds_name = "volumes/raw/s0"

    def __init__(self, image_size, transform=None) -> None:
        super().__init__()
        self.dataset = open_ds(self.container, self.ds_name)
        self.voxel_size = Coordinate(self.dataset.voxel_size)
        self.image_size = Coordinate(image_size) * self.voxel_size
        self.transform = transform

    def worker_init(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        print(f"Initializing worker {worker_id}.")
        # TODO define the indices for each worker here?

    def __getitem__(self, item) -> np.ndarray:
        x, y, z = super().__getitem__(item)
        center = Coordinate(z, y, x)
        corner = center - self.image_size // 2
        roi = Roi(corner, self.image_size)
        roi = roi.snap_to_grid(self.voxel_size, mode="closest")
        # Check if the roi is within the dataset
        if not self.dataset.roi.contains(roi):
            logger.warning(f"ROI {roi} at index {item} is outside the dataset.")
            # Remove the index from the list of indices
            # self.indices = np.delete(self.indices, item)
            # Try again
            return None
        data = self.dataset[roi].to_ndarray()
        assert data.shape == self.image_size / self.voxel_size, f"Expected {self.image_size}, got {data.shape}."
        data = Image.fromarray(data.squeeze()) 
        if self.transform:
            data = self.transform(data)
        return data 
        
