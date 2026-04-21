"""
ctypes bridge to librec_engine.so (compiled from MPI C sources).

Exposed C functions used here:
  Matrix  *load_matrix(const char *filename)
  void     free_matrix(Matrix *matrix)
  float   *compute_norms(const Matrix *matrix)
  UserRec *get_similar_users(float *sim_matrix, int user_id, int k, int num_users)
  ItemRec *get_item_recommendations(float *sim, float *ratings, int user_id,
                                     int k, int num_users, int num_items, int num_neighbors)

Cosine similarity is computed here in numpy: the C functions accept a pre-computed
float* similarity array from the caller, which we pass via ctypes pointer.
"""

import ctypes
import os
import tempfile
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── ctypes struct definitions ─────────────────────────────────────────────────

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


# ── Bridge ────────────────────────────────────────────────────────────────────

class RecEngineBridge:
    """Load librec_engine.so and provide a high-level Python interface."""

    LIB_PATH = os.getenv("REC_ENGINE_LIB", "/usr/local/lib/librec_engine.so")

    def __init__(self) -> None:
        self._lib = ctypes.CDLL(self.LIB_PATH)
        self._libc = ctypes.CDLL("libc.so.6")
        self._bind()

    def _bind(self) -> None:
        """Declare argtypes / restype for every C function we call."""
        lib = self._lib

        lib.load_matrix.argtypes = [ctypes.c_char_p]
        lib.load_matrix.restype = ctypes.POINTER(_Matrix)

        lib.free_matrix.argtypes = [ctypes.POINTER(_Matrix)]
        lib.free_matrix.restype = None

        lib.compute_norms.argtypes = [ctypes.POINTER(_Matrix)]
        lib.compute_norms.restype = ctypes.POINTER(ctypes.c_float)

        lib.get_similar_users.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # similarity_matrix (flattened)
            ctypes.c_int,                    # user_id
            ctypes.c_int,                    # k
            ctypes.c_int,                    # num_users
        ]
        lib.get_similar_users.restype = ctypes.POINTER(_UserRec)

        lib.get_item_recommendations.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # similarity_matrix
            ctypes.POINTER(ctypes.c_float),  # rating_matrix
            ctypes.c_int,                    # user_id
            ctypes.c_int,                    # k
            ctypes.c_int,                    # num_users
            ctypes.c_int,                    # num_items
            ctypes.c_int,                    # num_neighbors
        ]
        lib.get_item_recommendations.restype = ctypes.POINTER(_ItemRec)

        # libc free for C-allocated arrays
        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_from_csv(
        self, csv_path: str, top_k: int = 50
    ) -> Tuple[Dict[int, List[Dict]], np.ndarray]:
        """
        Load the rating matrix from csv_path, compute cosine similarity in numpy,
        then use the C library to extract top-k similar users per user.

        Returns:
            recommendations  – {user_id: [{"user_id": int, "score": float}, ...]}
            similarity_matrix – numpy float32 array shape (num_users, num_users)
        """
        # 1. Load rating matrix via C
        matrix_ptr = self._lib.load_matrix(csv_path.encode())
        if not matrix_ptr:
            raise RuntimeError(f"load_matrix returned NULL for {csv_path}")

        matrix = matrix_ptr.contents
        num_users: int = matrix.rows
        num_items: int = matrix.cols
        logger.info("Matrix loaded: %d users × %d items", num_users, num_items)

        # 2. Extract rating matrix as numpy array (copy before freeing C memory)
        n_elems = num_users * num_items
        rating_np = np.ctypeslib.as_array(matrix.data, shape=(n_elems,)).copy().astype(np.float32)
        rating_matrix = rating_np.reshape(num_users, num_items)

        # 3. Compute cosine similarity in numpy (avoids missing compute_similarity in .so)
        norms = np.linalg.norm(rating_matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        normalized = rating_matrix / norms
        similarity: np.ndarray = (normalized @ normalized.T).astype(np.float32)

        # 4. Free C matrix
        self._lib.free_matrix(matrix_ptr)

        # 5. Get contiguous flat arrays as ctypes pointers
        sim_flat = np.ascontiguousarray(similarity.flatten(), dtype=np.float32)
        rating_flat = np.ascontiguousarray(rating_matrix.flatten(), dtype=np.float32)

        sim_ptr = sim_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rating_ptr = rating_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # 6. Call get_similar_users for every user via C
        recommendations: Dict[int, List[Dict]] = {}
        actual_k = min(top_k, num_users - 1)

        for uid in range(num_users):
            recs_ptr = self._lib.get_similar_users(sim_ptr, uid, actual_k, num_users)
            if not recs_ptr:
                recommendations[uid] = []
                continue

            recs: List[Dict] = []
            for i in range(actual_k):
                rec = recs_ptr[i]
                if rec.user_id < 0:  # sentinel
                    break
                recs.append({"user_id": int(rec.user_id), "score": float(rec.similarity_score)})

            self._libc.free(ctypes.cast(recs_ptr, ctypes.c_void_p))
            recommendations[uid] = recs

        logger.info("Extracted recommendations for %d users (top_k=%d)", num_users, actual_k)
        return recommendations, similarity

    @staticmethod
    def write_interaction_csv(
        interactions: List[Tuple[int, str, float]],
        path: str,
    ) -> None:
        """
        Write [(user_id, product_uuid, rating), ...] to a CSV file the C loader expects.

        File format:  user_id,product_id,rating  (no header)
        """
        with open(path, "w") as fh:
            for user_id, product_id, rating in interactions:
                fh.write(f"{user_id},{product_id},{rating:.4f}\n")
