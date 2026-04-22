"""
Python wrapper for librec_engine.so (compiled from MPI C sources).

Actual exported symbols used:
  Matrix  *load_matrix(const char *filename)
  void     free_matrix(Matrix *matrix)
  UserRec *get_similar_users(float *sim_matrix, int user_id, int k, int num_users)
  ItemRec *get_item_recommendations(float *sim, float *ratings, int user_id,
                                     int k, int num_users, int num_items, int num_neighbors)

Cosine similarity is computed in numpy from the loaded rating matrix.
"""

import ctypes
import os
from typing import List, Optional, Tuple

import numpy as np


# ── ctypes struct mirrors ──────────────────────────────────────────────────────

class _Matrix(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
    ]


class _UserRec(ctypes.Structure):
    _fields_ = [
        ("user_id",          ctypes.c_int),
        ("similarity_score", ctypes.c_float),
    ]


class _ItemRec(ctypes.Structure):
    _fields_ = [
        ("item_id",          ctypes.c_int),
        ("predicted_rating", ctypes.c_float),
    ]


# ── Wrapper ───────────────────────────────────────────────────────────────────

class RecEngineWrapper:
    """High-level Python interface to the C recommendation engine."""

    def __init__(self, library_path: Optional[str] = None) -> None:
        if library_path is None:
            library_path = "/usr/local/lib/librec_engine.so"

        if not library_path or not os.path.exists(library_path):
            raise RuntimeError(
                f"Cannot find recommendation engine library at {library_path}\n"
                f"Build with: cd src/host-cuda/mpi/src && make librec_engine.so"
            )

        self._lib = ctypes.CDLL(library_path)
        self._libc = ctypes.CDLL("libc.so.6")
        self._setup_functions()

        self._initialized: bool = False
        self._num_users: int = 0
        self._num_items: int = 0
        # Stored as contiguous float32 numpy arrays to avoid repeated copies
        self._sim_flat: Optional[np.ndarray] = None     # (num_users * num_users,)
        self._rating_flat: Optional[np.ndarray] = None  # (num_users * num_items,)
        self._sim_matrix: Optional[np.ndarray] = None   # (num_users, num_users) view

    def _setup_functions(self) -> None:
        lib = self._lib

        lib.load_matrix.argtypes = [ctypes.c_char_p]
        lib.load_matrix.restype = ctypes.POINTER(_Matrix)

        lib.free_matrix.argtypes = [ctypes.POINTER(_Matrix)]
        lib.free_matrix.restype = None

        lib.get_similar_users.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.get_similar_users.restype = ctypes.POINTER(_UserRec)

        lib.get_item_recommendations.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.get_item_recommendations.restype = ctypes.POINTER(_ItemRec)

        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None

    def init(self, csv_path: str) -> bool:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        matrix_ptr = self._lib.load_matrix(csv_path.encode())
        if not matrix_ptr:
            return False

        matrix = matrix_ptr.contents
        num_users: int = matrix.rows
        num_items: int = matrix.cols
        n_elems = num_users * num_items

        # Copy rating data out of C-owned memory before freeing
        rating_np = np.ctypeslib.as_array(matrix.data, shape=(n_elems,)).copy().astype(np.float32)
        rating_matrix = rating_np.reshape(num_users, num_items)

        # Cosine similarity in numpy
        norms = np.linalg.norm(rating_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = rating_matrix / norms
        sim_matrix: np.ndarray = (normalized @ normalized.T).astype(np.float32)

        self._lib.free_matrix(matrix_ptr)

        # Keep contiguous flat copies so C pointers stay valid during calls
        self._sim_flat = np.ascontiguousarray(sim_matrix.flatten(), dtype=np.float32)
        self._rating_flat = np.ascontiguousarray(rating_np, dtype=np.float32)
        self._sim_matrix = sim_matrix
        self._num_users = num_users
        self._num_items = num_items
        self._initialized = True
        return True

    def get_similar_users(self, user_id: int, k: int = 10) -> List[int]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")

        actual_k = min(k, self._num_users - 1)
        sim_ptr = self._sim_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        recs_ptr = self._lib.get_similar_users(sim_ptr, user_id, actual_k, self._num_users)
        if not recs_ptr:
            return []

        results: List[int] = []
        for i in range(actual_k):
            rec = recs_ptr[i]
            if rec.user_id < 0:
                break
            results.append(int(rec.user_id))

        self._libc.free(ctypes.cast(recs_ptr, ctypes.c_void_p))
        return results

    def get_item_recommendations(
        self, user_id: int, k: int = 10, num_neighbors: int = 10
    ) -> List[int]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")

        actual_k = min(k, self._num_items)
        actual_neighbors = min(num_neighbors, self._num_users - 1)
        sim_ptr = self._sim_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rating_ptr = self._rating_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        recs_ptr = self._lib.get_item_recommendations(
            sim_ptr, rating_ptr, user_id, actual_k,
            self._num_users, self._num_items, actual_neighbors,
        )
        if not recs_ptr:
            return []

        results: List[int] = []
        for i in range(actual_k):
            rec = recs_ptr[i]
            if rec.item_id < 0:
                break
            results.append(int(rec.item_id))

        self._libc.free(ctypes.cast(recs_ptr, ctypes.c_void_p))
        return results

    def get_similarity(self, user_id_a: int, user_id_b: int) -> float:
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return float(self._sim_matrix[user_id_a, user_id_b])

    def get_dimensions(self) -> Tuple[int, int]:
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        return (self._num_users, self._num_items)

    def print_status(self) -> None:
        print(
            f"RecEngine: {self._num_users} users × {self._num_items} items, "
            f"initialized={self._initialized}"
        )

    def cleanup(self) -> None:
        self._sim_flat = None
        self._rating_flat = None
        self._sim_matrix = None
        self._initialized = False

    def __del__(self) -> None:
        if self._initialized:
            self.cleanup()


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine_instance: Optional[RecEngineWrapper] = None


def get_engine(library_path: Optional[str] = None) -> RecEngineWrapper:
    """Return the process-wide singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecEngineWrapper(library_path)
    return _engine_instance
