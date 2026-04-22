from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional
import numpy as np
from utils.errornorms import NormType

# Define metric types for history tracking
class MetricType(Enum):
    ABSOLUTE_ERROR = "absolute_error"   # ||u - u^k||
    RELATIVE_ERROR = "relative_error"   # ||u - u^k|| / ||u||
    CONVERGENCE_RATE = "convergence_rate"  # ||u - u^k|| / ||u - u^{k-1}||
    ITERATION_ERROR = "iteration_error"  # ||u^k - u^{k-1}||

# Define spatial and temporal modes for history tracking
class SpatialMode(str, Enum):
    """
    GLOBAL: store a single value per iteration that aggregates error across the entire domain.
    SUBDOMAINS: store separate values for each subdomain at each iteration, allowing for analysis of convergence behavior on individual subdomains.
    BOTH: store both global and subdomain metrics.
    """ 
    GLOBAL = "global"
    SUBDOMAINS = "subdomains"
    BOTH = "both"

class TemporalMode(str, Enum):
    """
    STATIC: no time dimension, store a single value per iteration (e.g. max error across all time steps)
    TIME: store values for each time step, resulting in a time series of metrics for each iteration.
    """
    STATIC = "static"
    TIME = "time"

# Define a dataclass for metric specifications
@dataclass
class MetricSpec:
    """
    Specification for a metric to be tracked during Schwarz iteration.
    - `name`: The type of metric to track (e.g. absolute error, relative error, convergence rate, iteration error).
    - `spatial`: Whether to store the metric as a global value (aggregated across the entire domain) or separately for each subdomain.
    - `temporal`: Whether to store the metric as a single value per iteration (e.g. max error across all time steps) or as a time series for each iteration.
    """
    name: MetricType
    spatial: SpatialMode
    temporal: TemporalMode

# Define a dataclass for history configuration
@dataclass
class HistoryConfig:
    """
    Configuration for tracking history of metrics during Schwarz iteration.
    - `metrics`: A list of MetricSpec objects specifying which metrics to track and how to store them.
    - `exact`: The exact solution function, used for computing error norms if required by the specified metrics.
    - `uh`: The reference solution (e.g. FEM solution), used for computing error norms if required by the specified metrics.
    - `time_indices`: Optional array of time step indices to track if temporal storage is enabled.
    - `subdomains`: Optional list of subdomain domainIDs to track if subdomain storage is enabled.
    - `mode`: The mode to use for error norm computation, either 'fem' to use the reference solution `uh` or 'exact' to use the exact solution `exact`.
    """
    metrics: list[MetricSpec]
    norm: NormType = NormType.L2
    exact: Optional[Callable] = None
    uh: Optional[np.ndarray] = None
    time_indices: Optional[np.ndarray] = None
    subdomains: Optional[list[int]] = None
    mode: Literal['fem', 'exact'] = 'fem'

    # Add a validation method to ensure that the provided configuration is consistent with the specified metrics
    def validate(self):
        counts = Counter(m.name for m in self.metrics)
        duplicates = [k for k, v in counts.items() if v > 1]
        assert not duplicates, f"Duplicate metrics found: {duplicates}"
        has_time = self.time_indices is not None and len(self.time_indices) > 0
        has_sub = self.subdomains is not None and len(self.subdomains) > 0
        for m in self.metrics:
            if m.temporal == TemporalMode.TIME:
                assert has_time, (f"Metric {m.name} requires time_indices, but none provided.")
            if m.spatial in (SpatialMode.SUBDOMAINS, SpatialMode.BOTH):
                assert has_sub, (f"Metric {m.name} requires subdomains, but none provided.")

# Define a dataclass for storing history values
@dataclass
class History:
    """
    Storage for history of metrics during Schwarz iteration.
    - `values`: A nested dictionary where the first key is the MetricType, the second key is either "global" or "subdomains" depending 
    on the spatial mode, and the value is either a numpy array or a dictionary of numpy arrays depending on the temporal mode and spatial mode:
        GLOBAL + STATIC        -> shape (niter,)
        GLOBAL + TIME          -> shape (niter, ntime)
        SUBDOMAINS + STATIC    -> dictionary of shape {domainID: shape (niter,)}
        SUBDOMAINS + TIME      -> dictionary of shape {domainID: shape (niter, ntime)}
    i.e. the structure of the `values` dictionary is:
    {
    MetricType.ITERATION_ERROR: {
        "global": np.array([...]), # shape (niter,) or shape (niter, ntime) depending on temporal mode                                                                                                                                      
        "subdomains": {domainID: np.array([...]), ...} # dictionary of shape {domainID: shape (niter,)} or {domainID: shape (niter, ntime)}
    },
    MetricType.ABSOLUTE_ERROR: {
        "global": np.array([...]), # shape (niter,) or shape (niter, ntime) depending on temporal mode
        "subdomains": {domainID: np.array([...]), ...} # dictionary of shape {domainID: shape (niter,)} or {domainID: shape (niter, ntime)}
    }
    - `time_indices`: Optional array of time step indices that were tracked.
    - `subdomains`: Optional list of subdomain domainIDs that were tracked.
    """
    values: dict[MetricType, dict[Literal["global", "subdomains"], np.ndarray | dict[int, np.ndarray]]]
    time_indices: Optional[np.ndarray]
    subdomains: Optional[list[int]]

# Define a function to initialize the history storage based on the provided configuration and number of iterations
def initialize_history(config: HistoryConfig) -> History:
    config.validate() # Ensure the provided configuration is valid
    values = {}
    for m in config.metrics:
        values[m.name] = {}
        if m.spatial in [SpatialMode.GLOBAL, SpatialMode.BOTH]:
            values[m.name]["global"] = []
        if m.spatial in [SpatialMode.SUBDOMAINS, SpatialMode.BOTH]:
            values[m.name]["subdomains"] = {}
            for domainID in config.subdomains: # type: ignore
                values[m.name]["subdomains"][domainID] = []
    return History(values = values, time_indices = config.time_indices, subdomains = config.subdomains)

# Define a function to record a metric value in the history storage based on the provided metric specification
def record(history: History, metric: MetricSpec, values: dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]):
    """
    Record a metric value in the history storage based on the provided metric specification.

    - `history`: The History object where the metric value should be stored.
    - `metric`: The MetricSpec object that specifies which metric is being recorded and how it should be stored.
    - `values`: The computed metric value(s) to be recorded. The structure of this dictionary should match the spatial 
    and temporal modes specified in the MetricSpec:
        GLOBAL + STATIC        -> scalar
        GLOBAL + TIME          -> (ntime,)
        SUBDOMAINS + STATIC    -> dictionary of shape {domainID: scalar}
        SUBDOMAINS + TIME      -> dictionary of shape {domainID: shape (len(time_indices),)}
    """
    assert metric.name in history.values, f"Metric {metric.name} not initialized in history."
    if metric.spatial in (SpatialMode.GLOBAL, SpatialMode.BOTH):
        history.values[metric.name]["global"].append(values["global"]) # type: ignore scalar or shape (ntime,)
    if metric.spatial in (SpatialMode.SUBDOMAINS, SpatialMode.BOTH):
        for domainID, sub_value in values["subdomains"].items(): # type: ignore dictionary of shape {domainID: scalar or shape (len(time_indices),)}
            history.values[metric.name]["subdomains"][domainID].append(sub_value) # type: ignore append scalar or shape (len(time_indices),) to dictionary of shape {domainID: list of scalar or list of shape (len(time_indices),)}

# Define a function to finalize the history storage by converting lists to numpy arrays for easier analysis and plotting
def finalize(history: History) -> History:
    """
    Finalize the history storage by converting lists to numpy arrays for easier analysis and plotting.
    - `history`: The History object containing the recorded metric values as lists.
    Returns a new History object with the same structure but with all metric values converted to numpy arrays. 
    The time_indices and subdomains are preserved as they are. The resulting History object will have metric 
    values stored as numpy arrays with shapes:
        GLOBAL + STATIC        -> (niter,)
        GLOBAL + TIME          -> (niter, ntime)
        SUBDOMAINS + STATIC    -> dictionary of shape {domainID: shape (niter,)}
        SUBDOMAINS + TIME      -> dictionary of shape {domainID: shape (niter, ntime)}
    """
    new_values = {}
    for m_name, modes in history.values.items():
        new_values[m_name] = {}
        for key, lst in modes.items():
            if key == "global":
                new_values[m_name][key] = np.array(lst) # shape (niter,) or shape (niter, ntime)
            elif key == "subdomains":                
                sub_values = {}
                for domainID, sub_lst in lst.items(): # type: ignore dictionary of shape {domainID: list of scalar or list of shape (len(time_indices),)}
                    sub_values[domainID] = np.array(sub_lst) # type: ignore shape (niter,) or shape (niter, ntime)
                new_values[m_name][key] = sub_values # type: ignore dictionary of shape {domainID: shape (niter,)} or dictionary of shape {domainID: shape (niter, ntime)}
    return History(values = new_values, time_indices = history.time_indices, subdomains = history.subdomains)