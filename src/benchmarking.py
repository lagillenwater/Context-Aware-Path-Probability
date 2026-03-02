"""
Benchmarking utilities for fair model comparison.

Provides standardized timing, memory tracking, and cost computation
to ensure fair comparison across different model architectures.

Key principle: All timing excludes I/O operations. Only computational
time is measured.

Standard node configuration:
- 4 CPU cores
- 32GB RAM
- Normalized cost = (time_hours) × (memory_gb / 32) × (cores / 4)
"""

import time
import psutil
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """
    Standardized benchmark result structure.

    Attributes:
        model_name: Name of model (e.g., 'Random_Forest')
        metapath: Metapath being modeled (e.g., 'CbGpPW')
        training_time_sec: Wall clock training time (excludes I/O)
        prediction_time_sec: Wall clock prediction time (excludes I/O)
        peak_memory_gb: Peak RSS memory usage
        n_cores_used: Number of cores used (always 4 for fair comparison)
        n_training_samples: Number of training samples
        n_parameters: Number of model parameters (if applicable)
        validation_r: Pearson correlation on validation set
        validation_mae: Mean absolute error on validation set
        validation_rmse: Root mean squared error on validation set
        normalized_cost: Standardized computational cost
        metadata: Additional model-specific information
    """
    model_name: str
    metapath: str
    training_time_sec: float
    prediction_time_sec: float
    peak_memory_gb: float
    n_cores_used: int
    n_training_samples: int
    n_parameters: Optional[int]
    validation_r: float
    validation_mae: float
    validation_rmse: float
    normalized_cost: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Recursively converts numpy types (float32, float64, int32, etc.) to
        native Python types to ensure JSON compatibility.

        Returns:
            Dictionary with all numpy types converted to Python types
        """
        result = asdict(self)
        return self._convert_numpy_types(result)

    @staticmethod
    def _convert_numpy_types(obj):
        """
        Recursively convert numpy types to native Python types.

        This ensures all numeric values can be serialized to JSON, avoiding
        TypeError when saving benchmark results that contain numpy arrays or
        scalar values from numpy operations.

        Args:
            obj: Object to convert (dict, list, numpy type, or primitive)

        Returns:
            Object with all numpy types converted to Python equivalents
        """
        if isinstance(obj, dict):
            return {key: BenchmarkResult._convert_numpy_types(value)
                    for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BenchmarkResult._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        else:
            # Fallback for other types
            return obj

    def save_json(self, filepath: Path):
        """Save benchmark to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Path) -> 'BenchmarkResult':
        """Load benchmark from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class Timer:
    """
    Context manager for timing code execution.

    Usage:
        with Timer() as t:
            # Code to time (excludes I/O)
            model.fit(X, y)

        elapsed_time = t.elapsed
    """

    def __init__(self):
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time


class MemoryTracker:
    """
    Track memory usage during model training/prediction.

    Usage:
        tracker = MemoryTracker()
        tracker.start()

        # Code to monitor
        model.fit(X, y)

        peak_memory_gb = tracker.stop()
    """

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0
        self.peak_memory = 0

    def start(self):
        """Start tracking memory."""
        self.baseline_memory = self.process.memory_info().rss / 1e9  # GB
        self.peak_memory = self.baseline_memory

    def update(self):
        """Update peak memory (call during long operations)."""
        current_memory = self.process.memory_info().rss / 1e9
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop(self) -> float:
        """
        Stop tracking and return peak memory.

        Returns:
            peak_memory_gb: Peak memory usage in GB
        """
        self.update()
        return self.peak_memory


def compute_normalized_cost(
    time_hours: float,
    memory_gb: float,
    cores: int,
    standard_cores: int = 4,
    standard_memory_gb: float = 32.0
) -> float:
    """
    Compute normalized computational cost.

    A "standard node-hour" is defined as:
    - 1 hour of computation
    - 4 CPU cores
    - 32GB RAM

    Cost scales linearly with time, memory, and cores.

    Args:
        time_hours: Computation time in hours
        memory_gb: Memory used in GB
        cores: Number of cores used
        standard_cores: Standard core count (default: 4)
        standard_memory_gb: Standard memory (default: 32.0)

    Returns:
        normalized_cost: Cost in standard node-hours

    Examples:
        >>> compute_normalized_cost(2.0, 64.0, 4)  # 2 hours, 64GB, 4 cores
        4.0  # 2× time × 2× memory = 4 standard node-hours

        >>> compute_normalized_cost(1.0, 16.0, 4)  # 1 hour, 16GB, 4 cores
        0.5  # 1× time × 0.5× memory = 0.5 standard node-hours
    """
    time_factor = time_hours
    memory_factor = memory_gb / standard_memory_gb
    core_factor = cores / standard_cores

    normalized_cost = time_factor * memory_factor * core_factor

    return normalized_cost


class ModelBenchmarker:
    """
    Comprehensive benchmarking for model training and validation.

    Automatically tracks:
    - Training time (excludes data loading)
    - Prediction time (excludes data loading)
    - Memory usage (peak RSS)
    - Computational cost (normalized)
    - Validation metrics (r, MAE, RMSE)

    Usage:
        benchmarker = ModelBenchmarker(
            model_name='Random_Forest',
            metapath='CbGpPW',
            n_training_samples=800
        )

        # Training
        with benchmarker.time_training():
            model.fit(X_train, y_train)

        # Prediction
        with benchmarker.time_prediction():
            predictions = model.predict(X_test)

        # Validation
        benchmarker.record_validation(predictions, y_test)

        # Save results
        result = benchmarker.finalize()
        result.save_json('results/benchmark.json')
    """

    def __init__(
        self,
        model_name: str,
        metapath: str,
        n_training_samples: int,
        n_cores: int = 4,
        n_parameters: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize benchmarker.

        Args:
            model_name: Name of the model
            metapath: Metapath being modeled
            n_training_samples: Number of training samples
            n_cores: Number of cores used (default: 4)
            n_parameters: Number of model parameters (optional)
            metadata: Additional metadata (optional)
        """
        self.model_name = model_name
        self.metapath = metapath
        self.n_training_samples = n_training_samples
        self.n_cores = n_cores
        self.n_parameters = n_parameters
        self.metadata = metadata or {}

        self.training_time_sec = 0.0
        self.prediction_time_sec = 0.0
        self.peak_memory_gb = 0.0

        self.validation_r = None
        self.validation_mae = None
        self.validation_rmse = None

        self.memory_tracker = MemoryTracker()

    def time_training(self) -> Timer:
        """
        Context manager for timing training.

        Usage:
            with benchmarker.time_training():
                model.fit(X_train, y_train)
        """
        return self._timing_context('training')

    def time_prediction(self) -> Timer:
        """
        Context manager for timing prediction.

        Usage:
            with benchmarker.time_prediction():
                predictions = model.predict(X_test)
        """
        return self._timing_context('prediction')

    def _timing_context(self, phase: str):
        """Internal timing context manager."""
        class TimingContext:
            def __init__(self, benchmarker, phase):
                self.benchmarker = benchmarker
                self.phase = phase
                self.timer = Timer()

            def __enter__(self):
                self.benchmarker.memory_tracker.start()
                self.timer.__enter__()
                return self

            def __exit__(self, *args):
                self.timer.__exit__(*args)
                peak_mem = self.benchmarker.memory_tracker.stop()

                if self.phase == 'training':
                    self.benchmarker.training_time_sec = self.timer.elapsed
                    self.benchmarker.peak_memory_gb = peak_mem
                elif self.phase == 'prediction':
                    self.benchmarker.prediction_time_sec = self.timer.elapsed
                    # Update peak memory if prediction uses more
                    self.benchmarker.peak_memory_gb = max(
                        self.benchmarker.peak_memory_gb,
                        peak_mem
                    )

        return TimingContext(self, phase)

    def record_validation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ):
        """
        Record validation metrics.

        Args:
            predictions: Predicted values
            actuals: Actual values
        """
        from scipy.stats import pearsonr

        # Pearson correlation
        self.validation_r, _ = pearsonr(predictions, actuals)

        # MAE
        self.validation_mae = np.mean(np.abs(predictions - actuals))

        # RMSE
        self.validation_rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    def finalize(self) -> BenchmarkResult:
        """
        Finalize benchmarking and return result.

        Returns:
            BenchmarkResult with all metrics
        """
        # Compute normalized cost
        time_hours = self.training_time_sec / 3600.0
        normalized_cost = compute_normalized_cost(
            time_hours,
            self.peak_memory_gb,
            self.n_cores
        )

        return BenchmarkResult(
            model_name=self.model_name,
            metapath=self.metapath,
            training_time_sec=self.training_time_sec,
            prediction_time_sec=self.prediction_time_sec,
            peak_memory_gb=self.peak_memory_gb,
            n_cores_used=self.n_cores,
            n_training_samples=self.n_training_samples,
            n_parameters=self.n_parameters,
            validation_r=self.validation_r,
            validation_mae=self.validation_mae,
            validation_rmse=self.validation_rmse,
            normalized_cost=normalized_cost,
            metadata=self.metadata
        )


def load_all_benchmarks(benchmark_dir: Path) -> list[BenchmarkResult]:
    """
    Load all benchmark results from a directory.

    Args:
        benchmark_dir: Directory containing benchmark JSON files

    Returns:
        List of BenchmarkResult objects
    """
    benchmarks = []

    for json_file in benchmark_dir.glob('*.json'):
        try:
            benchmark = BenchmarkResult.load_json(json_file)
            benchmarks.append(benchmark)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return benchmarks


def compute_efficiency(validation_r: float, normalized_cost: float) -> float:
    """
    Compute model efficiency (accuracy per unit cost).

    Args:
        validation_r: Validation correlation
        normalized_cost: Normalized computational cost

    Returns:
        efficiency: validation_r / normalized_cost
    """
    if normalized_cost == 0:
        return np.inf
    return validation_r / normalized_cost
