import numba as nb
import numpy as np

POINT_STORAGE_nb = nb.uint64
INDEX_STORAGE_nb = nb.uint32
FLOAT_STORAGE_nb = nb.float64

POINT_STORAGE_np = np.uint64
INDEX_STORAGE_np = np.uint32
FLOAT_STORAGE_np = np.float64

MAX_BCSR = 10
