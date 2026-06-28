
from __future__ import annotations

import ctypes
import json
import logging
import os
import struct
import subprocess
import tempfile
import time
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_OMP_LIB   = os.getenv("REC_ENGINE_LIB_OPENMP", "/usr/local/lib/librec_engine_openmp.so")
_CUDA_LIB  = os.getenv("REC_ENGINE_LIB_CUDA",   "/usr/local/lib/librec_engine_cuda.so")
_MPI_BENCH = os.getenv("MPI_BENCH_PATH",         "/usr/local/bin/similarity_mpi_bench")
_MPI_NP    = int(os.getenv("MPI_NPROCS", str(max(2, (os.cpu_count() or 4)))))

_LEGACY_LIB = os.getenv("REC_ENGINE_LIB", "/usr/local/lib/librec_engine.so")

class EngineType(str, Enum):
    OPENMP = "openmp"
    MPI    = "mpi"
    CUDA   = "cuda"

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

class _CtypesBridge:

    def __init__(self, lib_path: str, sim_fn_name: str) -> None:
        try:
            self._lib = ctypes.CDLL(lib_path)
        except OSError:
            if lib_path != _LEGACY_LIB:
                logger.warning("%s not found, falling back to %s", lib_path, _LEGACY_LIB)
                self._lib = ctypes.CDLL(_LEGACY_LIB)
            else:
                raise
        self._libc = ctypes.CDLL("libc.so.6")
        self._sim_fn_name = sim_fn_name
        self._bind()

    @staticmethod
    def _require_symbol(lib: ctypes.CDLL, name: str) -> None:

        fn = getattr(lib, name)
        addr = ctypes.cast(fn, ctypes.c_void_p).value
        if not addr:
            lib_name = getattr(lib, "_name", "<unknown>")
            raise AttributeError(
                f"Symbol '{name}' not found in {lib_name}. "
                "Ensure the correct .so was compiled and copied into the container."
            )

    def _bind(self) -> None:
        lib = self._lib

        lib.load_matrix.argtypes  = [ctypes.c_char_p]
        lib.load_matrix.restype   = ctypes.POINTER(_Matrix)

        lib.free_matrix.argtypes  = [ctypes.POINTER(_Matrix)]
        lib.free_matrix.restype   = None

        lib.compute_norms.argtypes = [ctypes.POINTER(_Matrix)]
        lib.compute_norms.restype  = ctypes.POINTER(ctypes.c_float)

        self._require_symbol(lib, self._sim_fn_name)
        sim_fn = getattr(lib, self._sim_fn_name)
        sim_fn.argtypes = [ctypes.POINTER(_Matrix), ctypes.POINTER(ctypes.c_float)]
        sim_fn.restype  = ctypes.POINTER(ctypes.c_float)
        self._sim_fn = sim_fn

        lib.get_similar_users.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.get_similar_users.restype = ctypes.POINTER(_UserRec)

        lib.get_item_recommendations.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.get_item_recommendations.restype = ctypes.POINTER(_ItemRec)

        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype  = None

    def compute_from_csv(
        self, csv_path: str, top_k: int = 50
    ) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict]:
        t0 = time.perf_counter()

        matrix_ptr = self._lib.load_matrix(csv_path.encode())
        if not matrix_ptr:
            raise RuntimeError(f"load_matrix returned NULL for {csv_path}")
        m = matrix_ptr.contents
        num_users, num_items = m.rows, m.cols
        logger.info("[%s] Matrix: %d×%d", self._sim_fn_name, num_users, num_items)

        norms_ptr = self._lib.compute_norms(matrix_ptr)

        t_load = time.perf_counter()

        sim_ptr = self._sim_fn(matrix_ptr, norms_ptr)
        if not sim_ptr:
            self._lib.free_matrix(matrix_ptr)
            self._libc.free(ctypes.cast(norms_ptr, ctypes.c_void_p))
            raise RuntimeError(f"{self._sim_fn_name} returned NULL")

        t_sim = time.perf_counter()

        sim_flat = np.ctypeslib.as_array(sim_ptr, shape=(num_users * num_users,)).copy()
        similarity = sim_flat.reshape(num_users, num_users).astype(np.float32)

        rating_np = np.ctypeslib.as_array(m.data, shape=(num_users * num_items,)).copy().astype(np.float32)

        self._libc.free(ctypes.cast(sim_ptr, ctypes.c_void_p))
        self._libc.free(ctypes.cast(norms_ptr, ctypes.c_void_p))
        self._lib.free_matrix(matrix_ptr)

        sim_c    = np.ascontiguousarray(similarity.flatten(), dtype=np.float32)
        rating_c = np.ascontiguousarray(rating_np,           dtype=np.float32)
        sim_cptr    = sim_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rating_cptr = rating_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        actual_k = min(top_k, num_users - 1)
        recommendations: Dict[int, List[Dict]] = {}

        for uid in range(num_users):
            recs_ptr = self._lib.get_similar_users(sim_cptr, uid, actual_k, num_users)
            if not recs_ptr:
                recommendations[uid] = []
                continue
            recs: List[Dict] = []
            for i in range(actual_k):
                rec = recs_ptr[i]
                if rec.user_id < 0:
                    break
                recs.append({"user_id": int(rec.user_id), "score": float(rec.similarity_score)})
            self._libc.free(ctypes.cast(recs_ptr, ctypes.c_void_p))
            recommendations[uid] = recs

        t_end = time.perf_counter()

        timing = {
            "engine":         self._sim_fn_name,
            "num_users":      num_users,
            "num_items":      num_items,
            "load_ms":        round((t_load - t0)  * 1000, 1),
            "similarity_ms":  round((t_sim  - t_load) * 1000, 1),
            "total_ms":       round((t_end  - t0)  * 1000, 1),
        }
        logger.info("[%s] Done — similarity %.0fms, total %.0fms",
                    self._sim_fn_name, timing["similarity_ms"], timing["total_ms"])
        return recommendations, similarity, timing

    @staticmethod
    def write_interaction_csv(interactions, path: str) -> None:
        with open(path, "w") as fh:
            for uid, pid, rating in interactions:
                fh.write(f"{uid},{pid},{float(rating):.4f}\n")

class OpenMPBridge(_CtypesBridge):

    def __init__(self) -> None:
        super().__init__(_OMP_LIB, "compute_similarity_omp")

class CUDABridge(_CtypesBridge):

    def __init__(self) -> None:
        super().__init__(_CUDA_LIB, "compute_similarity_cuda")

class MPIBridge:

    def __init__(self) -> None:
        if not os.path.isfile(_MPI_BENCH):
            raise FileNotFoundError(
                f"MPI bench binary not found at {_MPI_BENCH}. "
                "Ensure launch.sh compiled it and Dockerfile copied it."
            )

        try:
            self._rec_lib = ctypes.CDLL(_OMP_LIB)
        except OSError:
            self._rec_lib = ctypes.CDLL(_LEGACY_LIB)
        self._libc = ctypes.CDLL("libc.so.6")
        self._bind_rec_lib()

    def _bind_rec_lib(self) -> None:
        lib = self._rec_lib

        lib.get_similar_users.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.get_similar_users.restype = ctypes.POINTER(_UserRec)

        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype  = None

    def compute_from_csv(
        self, csv_path: str, top_k: int = 50
    ) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict]:
        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            sim_bin_path = tf.name

        try:
            cmd = ["mpirun", "--allow-run-as-root", "-np", str(_MPI_NP),
                   _MPI_BENCH, csv_path, sim_bin_path]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                logger.error("[MPI] mpirun failed:\n%s", result.stderr)
                raise RuntimeError(f"MPI bench exited {result.returncode}: {result.stderr[:500]}")

            timing_raw: Dict = {}
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.startswith("{"):
                    try:
                        timing_raw = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        pass

            num_users = timing_raw.get("num_users", 0)
            num_items = timing_raw.get("num_items", 0)

            with open(sim_bin_path, "rb") as fb:
                header = fb.read(8)
                if len(header) < 8:
                    raise RuntimeError("MPI bench produced empty output binary")
                nu, ni = struct.unpack("ii", header)
                num_users = num_users or nu
                sim_data = np.frombuffer(fb.read(), dtype=np.float32)

            if sim_data.size != num_users * num_users:
                raise RuntimeError(
                    f"Expected {num_users}×{num_users} floats, got {sim_data.size}"
                )
            similarity = sim_data.reshape(num_users, num_users).copy()

        finally:
            try:
                os.unlink(sim_bin_path)
            except OSError:
                pass

        t_end = time.perf_counter()

        sim_c    = np.ascontiguousarray(similarity.flatten(), dtype=np.float32)
        sim_cptr = sim_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        actual_k = min(top_k, num_users - 1)
        recommendations: Dict[int, List[Dict]] = {}

        for uid in range(num_users):
            recs_ptr = self._rec_lib.get_similar_users(sim_cptr, uid, actual_k, num_users)
            if not recs_ptr:
                recommendations[uid] = []
                continue
            recs: List[Dict] = []
            for i in range(actual_k):
                rec = recs_ptr[i]
                if rec.user_id < 0:
                    break
                recs.append({"user_id": int(rec.user_id), "score": float(rec.similarity_score)})
            self._libc.free(ctypes.cast(recs_ptr, ctypes.c_void_p))
            recommendations[uid] = recs

        timing = {
            "engine":        "mpi",
            "num_users":     num_users,
            "num_items":     num_items,
            "nprocs":        timing_raw.get("nprocs", _MPI_NP),
            "load_ms":       timing_raw.get("load_ms", 0),
            "bcast_ms":      timing_raw.get("bcast_ms", 0),
            "similarity_ms": timing_raw.get("similarity_ms", 0),
            "total_ms":      round((t_end - t0) * 1000, 1),
        }
        logger.info("[MPI] Done — similarity %.0fms (×%d procs), total %.0fms",
                    timing["similarity_ms"], timing["nprocs"], timing["total_ms"])
        return recommendations, similarity, timing

    @staticmethod
    def write_interaction_csv(interactions, path: str) -> None:
        with open(path, "w") as fh:
            for uid, pid, rating in interactions:
                fh.write(f"{uid},{pid},{float(rating):.4f}\n")

class NumpyFallbackBridge:

    def compute_from_csv(
        self, csv_path: str, top_k: int = 50
    ) -> Tuple[Dict[int, List[Dict]], np.ndarray, Dict]:
        t0 = time.perf_counter()

        interactions: List[Tuple[int, str, float]] = []
        with open(csv_path) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        interactions.append((int(parts[0]), parts[1], float(parts[2])))
                    except ValueError:
                        pass

        if not interactions:
            empty = np.zeros((0, 0), dtype=np.float32)
            return {}, empty, {"engine": "numpy_fallback", "num_users": 0, "num_items": 0, "total_ms": 0}

        user_ids = sorted({uid for uid, _, _ in interactions})
        item_ids = sorted({pid for _, pid, _ in interactions})
        user_idx = {u: i for i, u in enumerate(user_ids)}
        item_idx = {p: i for i, p in enumerate(item_ids)}
        N, M = len(user_ids), len(item_ids)

        matrix = np.zeros((N, M), dtype=np.float32)
        for uid, pid, rating in interactions:
            matrix[user_idx[uid], item_idx[pid]] = rating

        t_load = time.perf_counter()

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normalized = matrix / norms
        similarity = np.clip(normalized @ normalized.T, -1.0, 1.0).astype(np.float32)
        np.fill_diagonal(similarity, 1.0)

        t_sim = time.perf_counter()

        actual_k = min(top_k, N - 1)
        recommendations: Dict[int, List[Dict]] = {}
        for i in range(N):
            row = similarity[i].copy()
            row[i] = -2.0
            if actual_k > 0:
                top_idx = np.argpartition(row, -actual_k)[-actual_k:]
                top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
                recs = [{"user_id": int(j), "score": float(row[j])} for j in top_idx if row[j] > -2.0]
            else:
                recs = []
            recommendations[i] = recs

        t_end = time.perf_counter()

        timing = {
            "engine":        "numpy_fallback",
            "num_users":     N,
            "num_items":     M,
            "load_ms":       round((t_load - t0)   * 1000, 1),
            "similarity_ms": round((t_sim  - t_load) * 1000, 1),
            "total_ms":      round((t_end  - t0)   * 1000, 1),
        }
        logger.info("[numpy_fallback] Done — similarity %.0fms, total %.0fms",
                    timing["similarity_ms"], timing["total_ms"])
        return recommendations, similarity, timing

    @staticmethod
    def write_interaction_csv(interactions, path: str) -> None:
        with open(path, "w") as fh:
            for uid, pid, rating in interactions:
                fh.write(f"{uid},{pid},{float(rating):.4f}\n")

def create_bridge(
    engine_type: str | EngineType,
) -> OpenMPBridge | MPIBridge | CUDABridge | NumpyFallbackBridge:

    try:
        et = EngineType(str(engine_type).lower())
    except ValueError:
        logger.warning("Unknown engine '%s' — falling back to numpy", engine_type)
        return NumpyFallbackBridge()

    try:
        if et == EngineType.OPENMP:
            return OpenMPBridge()
        if et == EngineType.MPI:
            return MPIBridge()
        if et == EngineType.CUDA:
            return CUDABridge()
    except Exception as exc:
        logger.warning(
            "Failed to initialise %s engine (%s) — falling back to numpy", engine_type, exc
        )
        return NumpyFallbackBridge()

    return NumpyFallbackBridge()

class RecEngineBridge(OpenMPBridge):

    LIB_PATH = _OMP_LIB

    def compute_from_csv(self, csv_path: str, top_k: int = 50):  # type: ignore[override]
        recs, sim, _timing = super().compute_from_csv(csv_path, top_k)
        return recs, sim
