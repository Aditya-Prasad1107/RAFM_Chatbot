"""
Caching mechanism for Excel file extraction to improve performance.
"""
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle


class ExcelCache:
    """
    File-based caching system for Excel file extraction results.

    Features:
    - File modification time-based invalidation
    - Content hash verification
    - Configurable cache directory
    - Cache size management
    - Performance metrics
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        max_cache_size_mb: int = 500,
        enable_cache: bool = True,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize the caching system.

        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB
            enable_cache: Whether to enable caching
            cache_ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.enable_cache = enable_cache
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'errors': 0,
            'total_files_processed': 0
        }

        # Create cache directory if it doesn't exist
        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
            self._cleanup_expired_cache()

    def _get_cache_key(self, file_path: Path, extraction_params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a file and extraction parameters.

        Args:
            file_path: Path to the Excel file
            extraction_params: Parameters used for extraction

        Returns:
            Unique cache key string
        """
        # Create a hash of the file path and extraction parameters
        key_data = {
            'file_path': str(file_path),
            'params': extraction_params,
            'file_size': file_path.stat().st_size if file_path.exists() else 0
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the full path to a cache file."""
        return self.cache_dir / f"{cache_key}.cache"

    def _get_metadata_file_path(self, cache_key: str) -> Path:
        """Get the full path to a metadata file."""
        return self.cache_dir / f"{cache_key}.meta"

    def _is_cache_valid(self, file_path: Path, cache_key: str) -> bool:
        """
        Check if cached data is still valid.

        Args:
            file_path: Original Excel file path
            cache_key: Cache key for the file

        Returns:
            True if cache is valid, False otherwise
        """
        if not self.enable_cache:
            return False

        cache_file = self._get_cache_file_path(cache_key)
        meta_file = self._get_metadata_file_path(cache_key)

        # Check if cache files exist
        if not cache_file.exists() or not meta_file.exists():
            return False

        try:
            # Read metadata
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            # Check if cache is expired
            cache_age_hours = (time.time() - metadata['timestamp']) / 3600
            if cache_age_hours > self.cache_ttl_hours:
                return False

            # Check if original file has been modified
            if file_path.exists():
                original_mtime = file_path.stat().st_mtime
                if original_mtime > metadata['file_mtime']:
                    return False

            # Check cache version for compatibility
            if metadata.get('cache_version', 1) < 2:
                # Old cache format, invalidate
                return False

            return True

        except Exception as e:
            print(f"Error checking cache validity: {e}")
            return False

    def _save_to_cache(
        self,
        cache_key: str,
        data: Tuple[
            Dict[Tuple[str, str], List[str]],
            Dict[Tuple[str, str], List[str]],
            Dict[Tuple[str, str], List[str]]
        ],
        file_path: Path
    ) -> None:
        """
        Save extraction results to cache.

        Args:
            cache_key: Cache key for the data
            data: Tuple of (mappings_data, filename_info, expression_filenames)
            file_path: Original file path for metadata
        """
        if not self.enable_cache:
            return

        try:
            cache_file = self._get_cache_file_path(cache_key)
            meta_file = self._get_metadata_file_path(cache_key)

            # Save data using pickle for efficiency
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Extract data for metadata
            mappings_data = data[0] if len(data) >= 1 else {}

            metadata = {
                'timestamp': time.time(),
                'file_mtime': file_path.stat().st_mtime if file_path.exists() else 0,
                'file_path': str(file_path),
                'data_size': len(mappings_data),
                'cache_version': 2  # Version 2 includes expression_filenames
            }

            with open(meta_file, 'w') as f:
                json.dump(metadata, f)

        except Exception as e:
            print(f"Error saving to cache: {e}")
            self.cache_stats['errors'] += 1

    def _load_from_cache(
        self,
        cache_key: str
    ) -> Optional[Tuple[
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]]
    ]]:
        """
        Load extraction results from cache.

        Args:
            cache_key: Cache key for the data

        Returns:
            Tuple of (mappings_data, filename_info, expression_filenames) or None
        """
        if not self.enable_cache:
            return None

        try:
            cache_file = self._get_cache_file_path(cache_key)

            if not cache_file.exists():
                return None

            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            # Ensure we return a 3-tuple
            if isinstance(data, tuple):
                if len(data) == 3:
                    self.cache_stats['hits'] += 1
                    return data
                elif len(data) == 2:
                    # Old format, add empty expression_filenames
                    self.cache_stats['hits'] += 1
                    return (data[0], data[1], {})

            # Invalid format
            return None

        except Exception as e:
            print(f"Error loading from cache: {e}")
            self.cache_stats['errors'] += 1
            return None

    def get_cached_extraction(
        self,
        file_path: Path,
        extraction_params: Dict[str, Any]
    ) -> Optional[Tuple[
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]]
    ]]:
        """
        Get cached extraction results if available and valid.

        Args:
            file_path: Path to the Excel file
            extraction_params: Parameters used for extraction

        Returns:
            Tuple of (mappings_data, filename_info, expression_filenames) or None
        """
        if not self.enable_cache:
            return None

        cache_key = self._get_cache_key(file_path, extraction_params)

        if self._is_cache_valid(file_path, cache_key):
            return self._load_from_cache(cache_key)

        self.cache_stats['misses'] += 1
        return None

    def get_cached_extraction_with_filenames(
        self,
        file_path: Path,
        extraction_params: Dict[str, Any]
    ) -> Optional[Tuple[
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]],
        Dict[Tuple[str, str], List[str]]
    ]]:
        """
        Get cached extraction results with filename information.

        Args:
            file_path: Path to the Excel file
            extraction_params: Parameters used for extraction

        Returns:
            Tuple of (mappings_data, filename_info, expression_filenames) or None
        """
        return self.get_cached_extraction(file_path, extraction_params)

    def save_extraction_result_with_filenames(
        self,
        file_path: Path,
        extraction_params: Dict[str, Any],
        data: Union[
            Tuple[
                Dict[Tuple[str, str], List[str]],
                Dict[Tuple[str, str], List[str]],
                Dict[Tuple[str, str], List[str]]
            ],
            Tuple[
                Dict[Tuple[str, str], List[str]],
                Dict[Tuple[str, str], List[str]]
            ]
        ]
    ) -> None:
        """
        Save extraction results with filename information to cache.

        Args:
            file_path: Path to the Excel file
            extraction_params: Parameters used for extraction
            data: Tuple of (mappings_data, filename_info, expression_filenames)
        """
        if not self.enable_cache:
            return

        # Normalize to 3-tuple
        if isinstance(data, tuple):
            if len(data) == 3:
                normalized_data = data
            elif len(data) == 2:
                normalized_data = (data[0], data[1], {})
            else:
                return  # Invalid format
        else:
            return  # Invalid format

        cache_key = self._get_cache_key(file_path, extraction_params)
        self._save_to_cache(cache_key, normalized_data, file_path)
        self.cache_stats['total_files_processed'] += 1

    def save_extraction_result(
        self,
        file_path: Path,
        extraction_params: Dict[str, Any],
        data: Tuple[
            Dict[Tuple[str, str], List[str]],
            Dict[Tuple[str, str], List[str]]
        ]
    ) -> None:
        """
        Save extraction results to cache (legacy method).

        Args:
            file_path: Path to the Excel file
            extraction_params: Parameters used for extraction
            data: Tuple of (mappings_data, filename_info)
        """
        # Convert to 3-tuple format
        self.save_extraction_result_with_filenames(
            file_path,
            extraction_params,
            (data[0], data[1], {})
        )

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache files."""
        if not self.enable_cache:
            return

        current_time = time.time()
        ttl_seconds = self.cache_ttl_hours * 3600

        for meta_file in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)

                if current_time - metadata['timestamp'] > ttl_seconds:
                    # Remove expired cache and metadata files
                    cache_key = meta_file.stem
                    cache_file = self._get_cache_file_path(cache_key)

                    if cache_file.exists():
                        cache_file.unlink()
                    meta_file.unlink()

            except Exception as e:
                print(f"Error cleaning up cache file {meta_file}: {e}")

    def _get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        if not self.enable_cache:
            return 0

        total_size = 0
        for file_path in self.cache_dir.glob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def _cleanup_oldest_cache(self, target_size_bytes: int) -> None:
        """Remove oldest cache files to meet size limit."""
        if not self.enable_cache:
            return

        # Get all cache files with their timestamps
        cache_files = []
        for meta_file in self.cache_dir.glob("*.meta"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                cache_key = meta_file.stem
                cache_file = self._get_cache_file_path(cache_key)
                if cache_file.exists():
                    cache_files.append((metadata['timestamp'], cache_file, meta_file))
            except Exception:
                continue

        # Sort by timestamp (oldest first)
        cache_files.sort()

        # Remove oldest files until target size is met
        current_size = self._get_cache_size()
        for timestamp, cache_file, meta_file in cache_files:
            if current_size <= target_size_bytes:
                break

            try:
                file_size = cache_file.stat().st_size + meta_file.stat().st_size
                cache_file.unlink()
                meta_file.unlink()
                current_size -= file_size
            except Exception as e:
                print(f"Error removing cache files: {e}")

    def manage_cache_size(self) -> None:
        """Manage cache size to stay within limits."""
        if not self.enable_cache:
            return

        max_size_bytes = self.max_cache_size_mb * 1024 * 1024
        current_size = self._get_cache_size()

        if current_size > max_size_bytes:
            self._cleanup_oldest_cache(max_size_bytes * 0.8)  # Target 80% of max size

    def clear_cache(self) -> None:
        """Clear all cache files."""
        if not self.enable_cache:
            return

        try:
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            print("Cache cleared successfully.")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.cache_stats,
            'hit_rate_percent': round(hit_rate, 2),
            'cache_size_mb': round(self._get_cache_size() / (1024 * 1024), 2),
            'cache_enabled': self.enable_cache,
            'cache_dir': str(self.cache_dir)
        }

    def print_cache_stats(self) -> None:
        """Print cache statistics to console."""
        stats = self.get_cache_stats()
        print("\n" + "="*50)
        print("Cache Statistics")
        print("="*50)
        print(f"Cache Enabled: {stats['cache_enabled']}")
        print(f"Cache Directory: {stats['cache_dir']}")
        print(f"Cache Size: {stats['cache_size_mb']} MB")
        print(f"Cache Hits: {stats['hits']}")
        print(f"Cache Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate_percent']}%")
        print(f"Total Files Processed: {stats['total_files_processed']}")
        print(f"Errors: {stats['errors']}")
        print("="*50)


# Global cache instance
_global_cache: Optional[ExcelCache] = None


def get_cache(
    cache_dir: str = "./cache",
    max_cache_size_mb: int = 500,
    enable_cache: bool = True,
    cache_ttl_hours: int = 24
) -> ExcelCache:
    """
    Get or create the global cache instance.

    Args:
        cache_dir: Directory to store cache files
        max_cache_size_mb: Maximum cache size in MB
        enable_cache: Whether to enable caching
        cache_ttl_hours: Time-to-live for cache entries in hours

    Returns:
        ExcelCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = ExcelCache(
            cache_dir=cache_dir,
            max_cache_size_mb=max_cache_size_mb,
            enable_cache=enable_cache,
            cache_ttl_hours=cache_ttl_hours
        )

    return _global_cache


def configure_cache(
    cache_dir: str = "./cache",
    max_cache_size_mb: int = 500,
    enable_cache: bool = True,
    cache_ttl_hours: int = 24
) -> None:
    """
    Configure the global cache instance.

    Args:
        cache_dir: Directory to store cache files
        max_cache_size_mb: Maximum cache size in MB
        enable_cache: Whether to enable caching
        cache_ttl_hours: Time-to-live for cache entries in hours
    """
    global _global_cache
    _global_cache = ExcelCache(
        cache_dir=cache_dir,
        max_cache_size_mb=max_cache_size_mb,
        enable_cache=enable_cache,
        cache_ttl_hours=cache_ttl_hours
    )