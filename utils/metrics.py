from typing import Literal, Optional
import numpy as np
from fem.femspace import FEMSpace
from utils.errornorms import ErrorNorms, NormType
from utils.history import HistoryConfig, MetricSpec, MetricType, SpatialMode, TemporalMode
from utils.logger import get_logger
logger = get_logger(__name__)

def compute_many(error: ErrorNorms, norm: NormType, t_indices: list | np.ndarray) -> np.ndarray:
    """
    Compute the specified norm for multiple time indices and return an array of results. 
    This is a helper method to compute norms at multiple time steps when temporal storage 
    is enabled in the `HistoryConfig`.

    Parameters
    ----------
    error : ErrorNorms
        An instance of the ErrorNorms class initialized with the appropriate FEM space, solutions, and time grid.
    norm : NormType
        The type of norm to compute (e.g. NormType.L2, NormType.H1_SEMI, etc.)
    t_indices : list | np.ndarray
        A list or array of time step indices for which to compute the norm. Each index should be an integer 
        corresponding to a valid time step in the time grid. The method will compute the norm at each specified 
        time step and return an array of results.
    
    Returns
    -------
    np.ndarray, shape (len(t_indices),)
        An array containing the computed norm values for each specified time index. The length of the array will be 
        equal to the number of time indices provided. Each entry in the array corresponds to the computed norm at 
        the respective time index.
    """
    return np.array([error.compute(norm, t_index = t) for t in t_indices])

def compute_iteration_error(config: HistoryConfig, metric: MetricSpec, time_grid: np.ndarray,
                            gfemspace: FEMSpace, lfemspace: dict[int, FEMSpace],
                            current_ldata: dict[int, np.ndarray], prev_ldata: dict[int, np.ndarray], 
                            current_gdata: Optional[np.ndarray] = None, prev_gdata: Optional[np.ndarray] = None) -> dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]:
    """
    Compute the iteration error metric (||u^k - u^{k-1}||) for Schwarz iteration. This function computes the difference
    between the current and previous solutions for each subdomain, combines them into a global solution if necessary, 
    and then computes the specified norm of the iteration error according to the parameters in the provided 
    `HistoryConfig`. The computed iteration error can be returned as a single value for global storage or 
    as a dictionary of values for each subdomain if subdomain storage is enabled.

    Parameters
    ----------
    config : HistoryConfig
        The HistoryConfig instance containing the parameters for computing and storing the iteration error metric.
    metric : MetricSpec
        The MetricSpec object specifying the details of the iteration error metric to compute, including its name, spatial and temporal modes.
    gfemspace : FEMSpace
        The global FEM space used for computing the global iteration error if spatial storage includes global metrics.
    lfemspace : dict[int, FEMSpace]
        A dictionary mapping subdomain domainIDs to their respective FEM spaces, used for computing subdomain iteration errors if spatial storage includes subdomain metrics.
    time_grid : np.ndarray
        The array of time steps corresponding to the solutions.
    current_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their current local solution data (u^k) for the current Schwarz iteration.
    prev_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their previous local solution data (u^{k-1}) from the previous Schwarz iteration, used for computing the iteration error as the difference between current and previous solutions.
    current_gdata : Optional[np.ndarray]
        The current global solution data (u^k) for the current Schwarz iteration, used for computing the global iteration error if necessary.
    prev_gdata : Optional[np.ndarray]
        The previous global solution data (u^{k-1}) from the previous Schwarz iteration, used for computing the global iteration error if necessary.

    Returns
    -------
    dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]
        A dictionary containing the computed iteration error norms. The dictionary has two possible keys:
        - "global": The global iteration error norm(s), either a scalar or a numpy array of shape (len(time_indices),) depending on the temporal mode.
        - "subdomains": A dictionary mapping subdomain domainIDs to their respective iteration error norms, either a scalar or a numpy array of shape (len(time_indices),) depending on the temporal mode.
    """
    logger.debug(f"Computing iteration error metric for iteration...")

    # Initialize an empty dictionary to store the computed metric values for this iteration.
    values = {}

    # Compute the difference between current and previous solutions for each subdomain
    solution_diff = {domainID: current_ldata[domainID] - prev_ldata[domainID] for domainID in current_ldata}

    if metric.spatial in [SpatialMode.GLOBAL, SpatialMode.BOTH]:
        assert current_gdata is not None and prev_gdata is not None, "Global data must be provided for computing global iteration error."
        oswr_sol = current_gdata - prev_gdata
        est = ErrorNorms(femspace = gfemspace, u1 = oswr_sol, time = time_grid, mode = 'self')
        logger.debug(f"Computing global iteration error norm using {metric.norm} norm...")
        if metric.temporal == TemporalMode.STATIC:
            iter_error = est.compute(metric.norm) # scalar
        else:
            iter_error = compute_many(est, metric.norm, t_indices = config.time_indices) # type: ignore shape (len(t_indices),)
        values["global"] = iter_error
    if metric.spatial in [SpatialMode.SUBDOMAINS, SpatialMode.BOTH]:
        sub_errors = {}
        for domainID in config.subdomains: # type: ignore
            logger.debug(f"Computing iteration error norm for subdomain {domainID} using {metric.norm} norm...")
            sub_est = ErrorNorms(femspace = lfemspace[domainID], u1 = solution_diff[domainID], time = time_grid, mode = 'self')
            if metric.temporal == TemporalMode.STATIC:
                sub_error = sub_est.compute(metric.norm) # scalar
            else:
                sub_error = compute_many(sub_est, metric.norm, t_indices = config.time_indices) # type: ignore shape (len(t_indices),) 
            sub_errors[domainID] = sub_error
        values["subdomains"] = sub_errors # dictionary of shape {domainID: scalar or shape (len(t_indices),)}
    return values

def compute_absolute_error(config: HistoryConfig, metric: MetricSpec, time_grid: np.ndarray, ltog: dict[int, np.ndarray],
                           gfemspace: FEMSpace, lfemspace: dict[int, FEMSpace], current_ldata: dict[int, np.ndarray], 
                           current_gdata: Optional[np.ndarray] = None, mode: Literal['exact', 'fem'] = 'fem') -> dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]:
    """
    Compute the absolute error metric (||u_exact - u^k|| or ||u_h - u^k||) for Schwarz iteration. This function computes 
    the difference between the current solution and the exact solution (or the finite element solution) for each subdomain,
    combines them into a global solution if necessary, and then computes the specified norm of the absolute error according 
    to the parameters in the provided `HistoryConfig`. The computed absolute error can be returned as a single value for 
    global storage or as a dictionary of values for each subdomain if subdomain storage is enabled.

    Parameters
    ----------
    config : HistoryConfig
        The HistoryConfig instance containing the parameters for computing and storing the absolute error metric.
    metric : MetricSpec
        The MetricSpec object specifying the details of how to compute and store the absolute error metric, including its spatial and temporal modes.
    ltog : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their local-to-global index mappings, used for extracting the corresponding entries from the global 
        solution when computing the absolute error for subdomains if necessary.
    gfemspace : FEMSpace
        The global FEM space used for computing the global absolute error if spatial storage includes global metrics.
    lfemspace : dict[int, FEMSpace]
        A dictionary mapping subdomain domainIDs to their respective FEM spaces, used for computing subdomain absolute errors if spatial storage includes subdomain metrics.
    time_grid : np.ndarray
        The array of time steps corresponding to the solutions.
    current_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their current solution data (u^k) for the current Schwarz iteration.
    current_gdata : Optional[np.ndarray]
        The global solution data (u^k) for the current Schwarz iteration, used for computing the global absolute error if spatial storage includes global metrics.
    mode : Literal['exact', 'fem'], optional
        A string indicating whether to compute the absolute error with respect to the exact solution ('exact') or the finite element solution ('fem').

    Returns
    -------
    dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]
        A dictionary containing the computed absolute error norms. The dictionary has two possible keys:
        - "global": The global absolute error norm(s), either a scalar or a numpy array of shape (len(config.time_indices),) depending on the temporal mode.
        - "subdomains": A dictionary mapping subdomain domainIDs to their respective absolute error norms, either a scalar or a numpy array of shape (len(config.time_indices),) depending on the temporal mode.
    """
    logger.debug(f"Computing absolute error metric for iteration...")

    # Initialize an empty dictionary to store the computed metric values for this iteration.
    values = {}

    if metric.spatial in [SpatialMode.GLOBAL, SpatialMode.BOTH]:
        assert current_gdata is not None, "Global data must be provided for computing global absolute error."
        logger.debug(f"Computing global absolute error norm using {metric.norm} norm...")
        est = ErrorNorms(femspace = gfemspace, u1 = current_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = mode)
        if metric.temporal == TemporalMode.STATIC:
            abs_error = est.compute(metric.norm) 
        else:
            abs_error = compute_many(est, metric.norm, t_indices = config.time_indices) # type: ignore
        values["global"] = abs_error # shape (ntime,) or scalar
    if metric.spatial in [SpatialMode.SUBDOMAINS, SpatialMode.BOTH]:
        sub_errors = {}
        for domainID in config.subdomains: # type: ignore
            sub_est = ErrorNorms(femspace = lfemspace[domainID], u1 = current_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = mode)
            logger.debug(f"Computing iteration error norm for subdomain {domainID} using {metric.norm} norm...")
            if metric.temporal == TemporalMode.STATIC:
                sub_error = sub_est.compute(metric.norm) # scalar
            else:
                sub_error = compute_many(sub_est, metric.norm, t_indices = config.time_indices) # type: ignore shape (len(time_indices),)
            sub_errors[domainID] = sub_error
        values["subdomains"] = sub_errors # dictionary of shape {domainID: scalar or shape (len(time_indices),)}
    return values
    
# check this: this is not correct!
def compute_relative_error(config: HistoryConfig, metric: MetricSpec, time_grid: np.ndarray, ltog: dict[int, np.ndarray], 
                           gfemspace: FEMSpace, lfemspace: dict[int, FEMSpace], current_ldata: dict[int, np.ndarray], 
                           current_gdata: Optional[np.ndarray] = None, mode: Literal['exact', 'fem'] = 'fem') -> dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]:
    """
    Compute the relative error metric (||u_exact - u^k|| / ||u_exact|| or ||u_h - u^k|| / ||u_h||) for Schwarz iteration. 
    This function computes the absolute error as the difference between the current solution and the exact solution 
    (or the finite element solution) for each subdomain, combines them into a global solution if necessary, computes 
    the norm of the exact solution for normalization, and then computes the specified norm of the relative error according 
    to the parameters in the provided `HistoryConfig`. The computed relative error can be returned as a single value for 
    global storage or as a dictionary of values for each subdomain if subdomain storage is enabled.

    Parameters
    ----------
    config : HistoryConfig
        The HistoryConfig instance containing the parameters for computing and storing the relative error metric.
    metric : MetricSpec
        The MetricSpec object specifying the details of the relative error metric to compute, including its name, spatial and temporal modes.
    ltog : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their local-to-global index mappings, used for extracting the corresponding entries from the
        global solution when computing the relative error for subdomains if necessary.
    gfemspace : FEMSpace
        The global FEM space used for computing the global relative error if spatial storage includes global metrics.
    lfemspace : dict[int, FEMSpace]
        A dictionary mapping subdomain domainIDs to their respective FEM spaces, used for computing subdomain relative errors if spatial storage includes subdomain metrics.
    time_grid : np.ndarray
        The array of time steps corresponding to the solutions.
    current_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their current solution data (u^k) for the current Schwarz iteration.
    current_gdata : Optional[np.ndarray]
        The global solution data (u^k) for the current Schwarz iteration, used for computing the global absolute error if spatial storage includes global metrics.
    mode : Literal['exact', 'fem'], optional
        The mode for computing the relative error. 'exact' uses the exact solution, 'fem' uses the finite element solution.
    
    Returns
    -------
    dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]
        A dictionary containing the computed relative error norms. The dictionary has two possible keys:
        - "global": The global relative error norm(s), either a scalar or a numpy array of shape (len(time_indices),) depending on the temporal mode.
        - "subdomains": A dictionary mapping subdomain domainIDs to their respective relative error norms, either a scalar or a numpy array of shape (len(time_indices),) depending on the temporal mode.
    """
    logger.debug(f"Computing relative error metric for iteration...")
    
    # Initialize an empty dictionary to store the computed metric values for this iteration.
    values = {}

    if metric.spatial in [SpatialMode.GLOBAL, SpatialMode.BOTH]:
        assert current_gdata is not None, "Global data must be provided for computing global absolute error."
        est = ErrorNorms(femspace = gfemspace, u1 = current_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = mode)
        logger.debug(f"Computing global relative error norm using {metric.norm} norm...")
        if metric.temporal == TemporalMode.STATIC:
            num = est.compute(metric.norm)
            ref_est = ErrorNorms(femspace = gfemspace, u1 = current_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = 'self')
            den = ref_est.compute(metric.norm) + 1e-14
            rel_error = num / den
        else:
            num_array = compute_many(est, metric.norm, t_indices = config.time_indices) # type: ignore
            ref_est = ErrorNorms(femspace = gfemspace, u1 = current_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = 'self')
            den_array = compute_many(ref_est, metric.norm, t_indices = config.time_indices) + 1e-14 # type: ignore
            rel_error = num_array / den_array
        values["global"] = rel_error # shape (ntime,) or scalar
    if metric.spatial in [SpatialMode.SUBDOMAINS, SpatialMode.BOTH]:
        sub_errors = {}
        for domainID in config.subdomains: # type: ignore
            sub_est = ErrorNorms(femspace = lfemspace[domainID], u1 = current_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = 'auto')
            logger.debug(f"Computing relative error norm for subdomain {domainID} using {metric.norm} norm...")
            if metric.temporal == TemporalMode.STATIC:
                num = sub_est.compute(metric.norm)
                ref_est = ErrorNorms(femspace = lfemspace[domainID], u1 = current_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = 'self')
                den = ref_est.compute(metric.norm) + 1e-14
                sub_error = num / den
            else:
                num_array = compute_many(sub_est, metric.norm, t_indices = config.time_indices) # type: ignore (change below it is not correct)
                ref_est = ErrorNorms(femspace = lfemspace[domainID], u1 = current_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = 'self')
                den_array = compute_many(ref_est, metric.norm, t_indices = config.time_indices) + 1e-14 # type: ignore
                sub_error = num_array / den_array
            sub_errors[domainID] = sub_error
        values["subdomains"] = sub_errors # dictionary of shape {domainID: scalar or shape (len(time_indices),)}
    return values

def compute_convergence_rate(config: HistoryConfig, metric: MetricSpec, time_grid: np.ndarray, ltog: dict[int, np.ndarray], 
                            gfemspace: FEMSpace, lfemspace: dict[int, FEMSpace],
                            current_ldata: dict[int, np.ndarray], prev_ldata: dict[int, np.ndarray],
                            current_gdata: Optional[np.ndarray], prev_gdata: Optional[np.ndarray], 
                            mode: Literal['exact', 'fem'] = 'fem') -> dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]:
    """
    Compute the convergence rate metric (||u_exact - u^k|| / ||u_exact - u^{k-1}|| or ||u_h - u^k|| / ||u_h - u^{k-1}||) 
    for Schwarz iteration. This function computes the absolute error as the difference between the current solution and 
    the exact solution for each subdomain, computes the absolute error for the previous solution as well, combines them 
    into global solutions if necessary, computes the norms for normalization, and then computes the specified norm of 
    the convergence rate according to the parameters in the provided `HistoryConfig`. The computed convergence rate 
    can be returned as a single value for global storage or as a dictionary of values for each subdomain if subdomain storage is enabled.

    Parameters
    ----------
    config : HistoryConfig
        The HistoryConfig instance containing the parameters for computing and storing the convergence rate metric.
    metric : MetricSpec
        The MetricSpec object specifying the details of the convergence rate metric to compute, including its name, spatial and temporal modes.
    ltog : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their local-to-global index mappings, used for extracting the corresponding entries from the global 
        solution when computing the convergence rate for subdomains if necessary.
    gfemspace : FEMSpace
        The global FEM space used for computing the global convergence rate if spatial storage includes global metrics.
    lfemspace : dict[int, FEMSpace]
        A dictionary mapping subdomain domainIDs to their respective FEM spaces, used for computing subdomain convergence rates if spatial storage includes subdomain metrics.
    time_grid : np.ndarray
        The array of time steps corresponding to the solutions.
    current_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their current local solution data (u^k) for the current Schwarz iteration.
    current_gdata : Optional[np.ndarray]
        The current global solution data (u^k) for the current Schwarz iteration, if available.
    prev_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their previous local solution data (u^{k-1}) from the previous Schwarz iteration, used for computing the convergence rate as the ratio of the current absolute error to the previous absolute error.
    prev_gdata : Optional[np.ndarray]
        The previous global solution data (u^{k-1}) from the previous Schwarz iteration, if available.
    mode : Literal['exact', 'fem'], optional
        The mode for computing the convergence rate. 'exact' uses the exact solution, 'fem' uses the finite element solution.
    
    Returns
    -------
    dict
        A dictionary containing the computed convergence rates. The dictionary has two possible keys:
        - "global": The global convergence rate(s), either a scalar or a numpy array of shape (len(time_indices),) depending on the temporal mode.
        - "subdomains": A numpy array of shape (nsub,) or (nsub, len(time_indices)) containing the convergence rates for each subdomain, depending on the spatial and temporal modes.
    """  
    logger.debug(f"Computing convergence rate metric for iteration...")

    # Initialize an empty dictionary to store the computed metric values for this iteration.
    values = {}

    if metric.spatial in [SpatialMode.GLOBAL, SpatialMode.BOTH]:
        assert current_gdata is not None and prev_gdata is not None, "Global data must be provided for computing global convergence rate."
        est = ErrorNorms(femspace = gfemspace, u1 = current_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = mode)
        prev_est = ErrorNorms(femspace = gfemspace, u1 = prev_gdata, u2 = config.uh, u_exact = config.exact, time = time_grid, mode = mode)
        logger.debug(f"Computing global convergence rate using {metric.norm} norm...")
        if metric.temporal == TemporalMode.STATIC:
            num = est.compute(metric.norm)
            den = prev_est.compute(metric.norm) + 1e-14
            conv_rate = num / den
        else:
            num_array = compute_many(est, metric.norm, t_indices = config.time_indices) # type: ignore
            den_array = compute_many(prev_est, metric.norm, t_indices = config.time_indices) + 1e-14 # type: ignore
            conv_rate = num_array / den_array
        values["global"] = conv_rate # shape (ntime,) or scalar
    if metric.spatial in [SpatialMode.SUBDOMAINS, SpatialMode.BOTH]:
        sub_rates = {}
        for domainID in config.subdomains: # type: ignore
            sub_est = ErrorNorms(femspace = lfemspace[domainID], u1 = current_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = mode)
            sub_prev_est = ErrorNorms(femspace = lfemspace[domainID], u1 = prev_ldata[domainID], u2 = config.uh[ltog[domainID], :] if config.uh is not None else None, u_exact = config.exact, time = time_grid, mode = mode)
            logger.debug(f"Computing convergence rate for subdomain {domainID} using {metric.norm} norm...")
            if metric.temporal == TemporalMode.STATIC:
                num = sub_est.compute(metric.norm)
                den = sub_prev_est.compute(metric.norm) + 1e-14
                sub_rate = num / den
            else:
                num_array = compute_many(sub_est, metric.norm, t_indices = config.time_indices) # type: ignore
                den_array = compute_many(sub_prev_est, metric.norm, t_indices = config.time_indices) + 1e-14 # type: ignore
                sub_rate = num_array / den_array # shape (ntime,)
            sub_rates[domainID] = sub_rate
        values["subdomains"] = sub_rates  # dictionary of shape {domainID: scalar or shape (len(time_indices),)}
    return values
    
def compute_metrics(config: HistoryConfig, time_grid: np.ndarray, ltog: dict[int, np.ndarray], 
                    gfemspace: FEMSpace, lfemspace: dict[int, FEMSpace], 
                    current_ldata: dict[int, np.ndarray], prev_ldata: dict[int, np.ndarray], 
                    current_gdata: Optional[np.ndarray] = None, prev_gdata: Optional[np.ndarray] = None, 
                    mode: Literal['exact', 'fem'] = 'fem') -> dict[MetricSpec, dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]]:
    """
    Compute all metrics specified in the `HistoryConfig` for the current Schwarz iteration. This function iterates through the list of `MetricSpec` 
    objects in the `HistoryConfig`, determines which metrics need to be computed based on their types, and calls the appropriate computation functions 
    for each metric. The results are collected in a dictionary keyed by `MetricSpec`, where each value is itself a dictionary containing the computed 
    metric values for global and/or subdomain storage as specified in the `MetricSpec`. The computed metrics can include iteration error, absolute error, 
    relative error, and convergence rate, and they can be stored in various combinations of spatial and temporal modes as defined in the `HistoryConfig`. 
    The returned dictionary provides a structured way to access all the computed metrics for the current iteration.

    Parameters
    ----------
    config : HistoryConfig
        The HistoryConfig instance containing the list of MetricSpec objects that specify which metrics to compute and how to compute and store them.
    ltog : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their local-to-global index mappings, used for extracting the corresponding entries from the global 
        solution when computing metrics for subdomains if necessary.
    gfemspace : FEMSpace
        The global FEM space used for computing global metrics if spatial storage includes global metrics.
    lfemspace : dict[int, FEMSpace]
        A dictionary mapping subdomain domainIDs to their respective FEM spaces, used for computing subdomain metrics if spatial storage includes subdomain metrics.
    time_grid : np.ndarray
        The array of time steps corresponding to the solutions.
    current_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their current solution data (u^k) for the current Schwarz iteration, used as input for computing the metrics.
    prev_ldata : dict[int, np.ndarray]
        A dictionary mapping subdomain domainIDs to their previous solution data (u^{k-1}) from the previous Schwarz iteration, used as input for computing iteration error and convergence rate metrics that require the previous solution for comparison.
    current_gdata : Optional[np.ndarray]
        The global solution data for the current Schwarz iteration, used for computing global metrics if spatial storage includes global metrics.
    prev_gdata : Optional[np.ndarray]
        The global solution data from the previous Schwarz iteration, used for computing global metrics if spatial storage includes global metrics.
    mode : Literal['exact', 'fem'], optional
        The mode for computing the relative error. 'exact' uses the exact solution, 'fem' uses the finite element solution.

    Returns
    -------
    dict[MetricSpec, dict[Literal["global", "subdomains"], float | np.ndarray | dict[int, float | np.ndarray]]]
        A dictionary where each key is a MetricSpec corresponding to a metric specified in the HistoryConfig, and each value is a dictionary containing 
        the computed metric values for that MetricSpec. The inner dictionary provides a structured way to access all the computed metrics for the current iteration.
        For example, the returned dictionary may look like:
        {
            metric_spec: {
                "global": np.array([...]) or scalar,
                "subdomains": {domainID: np.array([...]) or scalar, ...}
            },
            ...
        }
    """
    assert mode in ['exact', 'fem'], "Mode must be either 'exact' or 'fem'."
    results = {}
    for metric_spec in config.metrics:
        if metric_spec.name == MetricType.ITERATION_ERROR:
            results[metric_spec] = compute_iteration_error(config, metric_spec, time_grid, gfemspace, lfemspace,
                                                                          current_ldata, prev_ldata, current_gdata, prev_gdata)
        elif metric_spec.name == MetricType.ABSOLUTE_ERROR:
            results[metric_spec] = compute_absolute_error(config, metric_spec, time_grid, ltog, gfemspace, lfemspace,
                                                                        current_ldata, current_gdata, mode)
        elif metric_spec.name == MetricType.RELATIVE_ERROR:
            results[metric_spec] = compute_relative_error(config, metric_spec, time_grid, ltog, gfemspace, lfemspace, 
                                                                        current_ldata, current_gdata, mode)
        elif metric_spec.name == MetricType.CONVERGENCE_RATE:
            results[metric_spec] = compute_convergence_rate(config, metric_spec, time_grid, ltog, gfemspace, lfemspace, 
                                                                            current_ldata, prev_ldata, current_gdata, prev_gdata, mode)
    return results