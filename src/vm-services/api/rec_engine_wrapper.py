"""
Recommendation Engine Python Wrapper
Uses ctypes to interface with the optimized C implementation
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class RecEngineWrapper:
    """High-level Python interface to the C recommendation engine"""

    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the recommendation engine wrapper
        
        Args:
            library_path: Path to compiled C library (.so file).
                         If None, searches in common locations.
        """
        if library_path is None:
            library_path = self._find_library()
        
        if not library_path or not os.path.exists(library_path):
            raise RuntimeError(
                f"Cannot find recommendation engine library at {library_path}\n"
                f"Please build the C code first using: cd src/host-cuda/openmp/src && make openmp"
            )
        
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load library {library_path}: {e}")
        
        # Define function signatures
        self._setup_functions()
        self._initialized = False
    
    def _find_library(self) -> Optional[str]:
        """Search for the compiled recommendation engine library"""
        # Search in common build locations
        search_paths = [
            "./lib/librec_engine.so",
            "../lib/librec_engine.so",
            "../../lib/librec_engine.so",
            "./src/host-cuda/openmp/src/bin/similarity_openmp",
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        
        return None
    
    def _setup_functions(self):
        """Setup C function signatures"""
        # rec_engine_init
        self.lib.rec_engine_init.argtypes = [ctypes.c_char_p]
        self.lib.rec_engine_init.restype = ctypes.c_int
        
        # rec_engine_get_similar_users
        self.lib.rec_engine_get_similar_users.argtypes = [
            ctypes.c_int, ctypes.c_int
        ]
        self.lib.rec_engine_get_similar_users.restype = ctypes.POINTER(ctypes.c_int)
        
        # rec_engine_get_item_recommendations
        self.lib.rec_engine_get_item_recommendations.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.rec_engine_get_item_recommendations.restype = ctypes.POINTER(ctypes.c_int)
        
        # rec_engine_get_similarity
        self.lib.rec_engine_get_similarity.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.rec_engine_get_similarity.restype = ctypes.c_float
        
        # rec_engine_get_dimensions
        self.lib.rec_engine_get_dimensions.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.rec_engine_get_dimensions.restype = None
        
        # rec_engine_cleanup
        self.lib.rec_engine_cleanup.argtypes = []
        self.lib.rec_engine_cleanup.restype = None
        
        # rec_engine_free_array
        self.lib.rec_engine_free_array.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.rec_engine_free_array.restype = None
        
        # rec_engine_print_status
        self.lib.rec_engine_print_status.argtypes = []
        self.lib.rec_engine_print_status.restype = None
    
    def init(self, csv_path: str) -> bool:
        """
        Initialize the recommendation engine with data
        
        Args:
            csv_path: Path to CSV file with columns: user_id, product_id, rating
            
        Returns:
            True on success, False on failure
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        result = self.lib.rec_engine_init(csv_path.encode('utf-8'))
        self._initialized = result == 1
        return self._initialized
    
    def get_similar_users(self, user_id: int, k: int = 10) -> List[int]:
        """
        Get k most similar users
        
        Args:
            user_id: Target user ID
            k: Number of similar users to return
            
        Returns:
            List of user IDs sorted by similarity
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        
        c_array = self.lib.rec_engine_get_similar_users(user_id, k)
        if not c_array:
            return []
        
        results = []
        for i in range(k):
            if c_array[i] == -1:  # Sentinel
                break
            results.append(c_array[i])
        
        self.lib.rec_engine_free_array(c_array)
        return results
    
    def get_item_recommendations(self, user_id: int, k: int = 10,
                                num_neighbors: int = 10) -> List[int]:
        """
        Get k item recommendations for a user
        
        Args:
            user_id: Target user ID
            k: Number of recommendations
            num_neighbors: Number of similar users to use in prediction
            
        Returns:
            List of item IDs sorted by predicted rating
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        
        c_array = self.lib.rec_engine_get_item_recommendations(
            user_id, k, num_neighbors
        )
        if not c_array:
            return []
        
        results = []
        for i in range(k):
            if c_array[i] == -1:  # Sentinel
                break
            results.append(c_array[i])
        
        self.lib.rec_engine_free_array(c_array)
        return results
    
    def get_similarity(self, user_id_a: int, user_id_b: int) -> float:
        """Get similarity score between two users"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        
        return self.lib.rec_engine_get_similarity(user_id_a, user_id_b)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get (num_users, num_items)"""
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call init() first.")
        
        num_users = ctypes.c_int()
        num_items = ctypes.c_int()
        self.lib.rec_engine_get_dimensions(
            ctypes.byref(num_users), ctypes.byref(num_items)
        )
        return (num_users.value, num_items.value)
    
    def print_status(self):
        """Print engine status"""
        self.lib.rec_engine_print_status()
    
    def cleanup(self):
        """Cleanup and free resources"""
        self.lib.rec_engine_cleanup()
        self._initialized = False
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        if self._initialized:
            self.cleanup()


# Singleton instance
_engine_instance: Optional[RecEngineWrapper] = None


def get_engine(library_path: Optional[str] = None) -> RecEngineWrapper:
    """Get or create singleton engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecEngineWrapper(library_path)
    return _engine_instance
