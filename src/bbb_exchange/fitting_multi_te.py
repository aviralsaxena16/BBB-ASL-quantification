import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from model_multi_te import deltaM_multite_model
import stan

import os

def load_stan_model(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stan_path = os.path.join(base_dir, "models", "stan", filename)

    if not os.path.exists(stan_path):
        raise FileNotFoundError(f"Stan file not found: {stan_path}")

    with open(stan_path, "r") as f:
        return f.read()


STAN_MODEL_CODE = load_stan_model("multite_full.stan")

import json
with open("config.json", "r") as file:
		config = json.load(file)

def bayesian_fit_voxel_multite(tis, tes, ntes, signal, M0a, taus,
                               param_config, att_ls_val=None, cbf_ls_val=None):
    """
    Bayesian fitting for one voxel using multi_te_model.py
    """
    import warnings

    # Validate inputs
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print("Invalid signal data")
        return None

    if len(signal) == 0:
        print("Empty signal data")
        return None

    # Set up priors based on LS results or defaults
    if att_ls_val is not None and np.isfinite(att_ls_val):
        att_prior_mean = float(np.clip(att_ls_val, 0.1, 3.0))
        att_prior_std = float(param_config.get('att_prior_std', 0.3))
    else:
        att_prior_mean = 0.8
        att_prior_std = 0.3

    if cbf_ls_val is not None and np.isfinite(cbf_ls_val):
        cbf_prior_mean = float(np.clip(cbf_ls_val, 10.0, 200.0))
        cbf_prior_std = float(param_config.get('cbf_prior_std', 15.0))
    else:
        cbf_prior_mean = 60.0
        cbf_prior_std = 20.0


    # Prepare data for Stan
    data = {
        "n_measurements": len(signal),
        "n_ti": len(tis),
        "signal": signal.astype(float),
        "tis": np.asarray(tis, float),
        "ntes": np.asarray(ntes, np.int32),
        "tes": np.asarray(tes, float),
        "tau_per_ti": (np.full(len(tis), taus, float)
                       if np.isscalar(taus) else np.asarray(taus, float)),

        "t1": param_config.get("T1", 1.3),
        "t1b": param_config.get("T1a", 1.65),
        "t2": param_config.get("T2", 0.050),
        "t2b": param_config.get("T2a", 0.150),
        "texch": param_config.get("texch", 0.1),
        "itt": param_config.get("itt", 0.2),
        "lambd": param_config.get("lambd", 0.9),
        "alpha": param_config.get("alpha", 0.68),
        "M0a": M0a,

        # Priors
        "att_prior_mean": att_prior_mean,
        "att_prior_std": att_prior_std,
        "cbf_prior_mean": cbf_prior_mean,
        "cbf_prior_std": cbf_prior_std,
    }

    try:
        print("Building Stan model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = stan.build(STAN_MODEL_CODE, data=data)

        if model is None:
            print("Stan model building failed")
            return None

        print("Sampling from Stan model...")
        fit = model.sample(num_chains=2, num_samples=500, num_warmup=200)

        if fit is None:
            print("Stan sampling failed")
            return None

        print("Bayesian fitting successful!")
        return fit

    except Exception as e:
        print(f"Error in Stan fitting: {e}")
        return None



def bayesian_fit_volume_multite(pwi_data, tis, tes, ntes, m0_data, taus, param_config,
                                att_ls_map=None, cbf_ls_map=None, max_voxels=None):
    """
    Bayesian fitting for entire volume
    """

    shape = pwi_data.shape[:3]

    # Initialize result maps
    att_map = np.full(shape, np.nan)
    cbf_map = np.full(shape, np.nan)
    att_std_map = np.full(shape, np.nan)
    cbf_std_map = np.full(shape, np.nan)

    total_voxels = shape[0] * shape[1] * shape[2]
    processed_voxels = 0
    successful_fits = 0

    if max_voxels is not None:
        total_voxels = min(total_voxels, max_voxels)

    print(f"Starting Multi-TE Bayesian fitting for up to {total_voxels} voxels...")

    voxel_count = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if max_voxels is not None and voxel_count >= max_voxels:
                    break

                processed_voxels += 1
                voxel_count += 1

                # Progress update
                if processed_voxels % 10 == 0:
                    print(f"Processing voxel {processed_voxels}/{total_voxels} "
                          f"({100 * processed_voxels / total_voxels:.1f}%) - "
                          f"Successful fits: {successful_fits}")

                signal = pwi_data[x, y, z, :]
                m0 = m0_data[x, y, z]

                # Check for invalid signal data
                if (np.any(np.isnan(signal)) or
                        np.any(np.isinf(signal)) or
                        np.all(signal == 0)):
                    continue

                # Check for invalid M0 data
                if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
                    continue

                # Normalize signal
                signal_normalized = signal / (m0 * 5)
                M0a = m0 / (6000 * 0.9)

                # Get LS prior values if available
                att_ls_val = att_ls_map[x, y, z] if att_ls_map is not None else None
                cbf_ls_val = cbf_ls_map[x, y, z] if cbf_ls_map is not None else None

                try:
                    # Fit this voxel
                    fit = bayesian_fit_voxel_multite(
                        tis, tes, ntes, signal_normalized, M0a, taus,
                        param_config, att_ls_val, cbf_ls_val
                    )

                    if fit is not None:
                        # Extract results
                        df = fit.to_frame()
                        att_samples = df['att'].values
                        cbf_samples = df['cbf'].values

                        att_map[x, y, z] = np.mean(att_samples)
                        cbf_map[x, y, z] = np.mean(cbf_samples)
                        att_std_map[x, y, z] = np.std(att_samples)
                        cbf_std_map[x, y, z] = np.std(cbf_samples)

                        successful_fits += 1

                except Exception as e:
                    print(f"Error fitting voxel ({x}, {y}, {z}): {e}")
                    continue

            if max_voxels is not None and voxel_count >= max_voxels:
                break
        if max_voxels is not None and voxel_count >= max_voxels:
            break

    print(f"Multi-TE Bayesian fitting completed: {successful_fits}/{processed_voxels} successful fits "
          f"({100 * successful_fits / processed_voxels:.1f}%)")

    # Prepare results dictionary
    results = {
        'att_map': att_map,
        'cbf_map': cbf_map,
        'att_std_map': att_std_map,
        'cbf_std_map': cbf_std_map,
        'successful_fits': successful_fits,
        'total_processed': processed_voxels
    }

    return results


def create_multite_bayesian_config():
    """
    Create Multi-TE specific Bayesian configuration
    """
    config = {
        # Fixed physiological parameters (same as LS fitting)
        'T1': 1.3,
        'T1a': 1.65,
        'T2': 0.050,
        'T2a': 0.150,
        'texch': 0.1,
        'itt': 0.2,
        'lambd': 0.9,
        'alpha': 0.68,

        # ATT and CBF priors from LS fitting
        'att_prior_from_ls': True,
        'cbf_prior_from_ls': True,
        'att_prior_std': 0.3,
        'cbf_prior_std': 15.0,

        # Processing options
        'max_voxels': 50,  # Reduced for testing
        'save_fitted_params': False  # Simplified - no additional parameter fitting
    }
    return config


def ls_fit_voxel_multite(tis, tes, ntes, signal, M0a, taus,
                         T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                         texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):
    """
	Least squares fitting for a single voxel using Multi-TE ASL model

	Parameters:
	- tis: array of inversion times [s]
	- tes: array of echo times [s]
	- ntes: array of number of TEs per TI
	- signal: measured signal values
	- M0a: arterial M0 scaling factor
	- taus: labeling duration(s) [s]
	- T1, T1a, T2, T2a: relaxation times [s]
	- texch: exchange time [s]
	- itt: intra-voxel transit time [s]
	- lambd: blood-tissue partition coefficient
	- alpha: labeling efficiency

	Returns:
	- att: fitted arterial transit time [s]
	- cbf: fitted cerebral blood flow [ml/min/100g]
	- rmse: root mean square error
	"""

    def model_func(combined_input, att, cbf):

        # Map parameter names to match deltaM_multite_model function signature
        return deltaM_multite_model(
            tis, tes, ntes, att, cbf, M0a, taus,
            t1=T1, t1b=T1a, t2=T2, t2b=T2a,
            texch=texch, itt=itt, lambd=lambd, alpha=alpha
        )

    param0 = [1.2, 60.0]
    bounds = ([0.1, 10.0], [2.5, 250.0])  # Bounds for att and cbf

    # Create dummy x data (curve_fit requires x data)
    x_dummy = np.arange(len(signal))

    try:
        param_opt, param_cov = curve_fit(
            lambda x, att, cbf: model_func(x, att, cbf),
            x_dummy, signal,
            p0=param0, bounds=bounds,
            maxfev=5000
        )

        att_fit, cbf_fit = param_opt

        # Calculate RMSE
        fitted_signal = deltaM_multite_model(
            tis, tes, ntes, att_fit, cbf_fit, M0a, taus,
            t1=T1, t1b=T1a, t2=T2, t2b=T2a,
            texch=texch, itt=itt, lambd=lambd, alpha=alpha
        )

        rmse = np.sqrt(np.mean((signal - fitted_signal) ** 2))

        return att_fit, cbf_fit, rmse

    except RuntimeError as e:
        print(f"Fitting failed: {e}")
        return np.nan, np.nan, np.nan


def ls_fit_volume_multite(pwi_data, tis, tes, ntes, m0_data, taus,
                          T1=1.3, T1a=1.65, T2=0.050, T2a=0.150,
                          texch=0.1, itt=0.2, lambd=0.9, alpha=0.68):
    """
	Least squares fitting for entire volume using Multi-TE ASL model

	Parameters:
	- pwi_data: 4D PWI data (x, y, z, time_points)
	- tis: array of inversion times [s]
	- tes: array of echo times [s]
	- ntes: array of number of TEs per TI
	- m0_data: 3D M0 data
	- taus: labeling duration(s) [s]
	- T1, T1a, T2, T2a: relaxation times [s]
	- texch: exchange time [s]
	- itt: intra-voxel transit time [s]
	- lambd: blood-tissue partition coefficient
	- alpha: labeling efficiency

	Returns:
	- att_map: 3D ATT map [s]
	- cbf_map: 3D CBF map [ml/min/100g]
	- rmse_map: 3D RMSE map
	"""

    shape = pwi_data.shape[:3]
    att_map = np.full(shape, np.nan)
    cbf_map = np.full(shape, np.nan)
    rmse_map = np.full(shape, np.nan)

    total_voxels = shape[0] * shape[1] * shape[2]
    processed_voxels = 0
    successful_fits = 0

    print(f"Starting Multi-TE LS fitting for {total_voxels} voxels...")

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                processed_voxels += 1

                # Progress update
                if processed_voxels % 1000 == 0:
                    print(f"Processing voxel {processed_voxels}/{total_voxels} "
                          f"({100 * processed_voxels / total_voxels:.1f}%) - "
                          f"Successful fits: {successful_fits}")

                signal = pwi_data[x, y, z, :]
                m0 = m0_data[x, y, z]

                # Check for invalid signal data
                if (np.any(np.isnan(signal)) or
                        np.any(np.isinf(signal)) or
                        np.all(signal == 0)):
                    continue

                # Check for invalid M0 data
                if (np.isnan(m0) or np.isinf(m0) or m0 <= 0):
                    continue

                # Normalize signal
                signal_normalized = signal / (m0 * 5)
                M0a = m0 / (6000 * 0.9)

                try:
                    att, cbf, rmse = ls_fit_voxel_multite(
                        tis, tes, ntes, signal_normalized, M0a, taus,
                        T1=T1, T1a=T1a, T2=T2, T2a=T2a,
                        texch=texch, itt=itt, lambd=lambd, alpha=alpha
                    )

                    # Store results if fitting was successful
                    if not (np.isnan(att) or np.isnan(cbf) or np.isnan(rmse)):
                        att_map[x, y, z] = att
                        cbf_map[x, y, z] = cbf
                        rmse_map[x, y, z] = rmse
                        successful_fits += 1

                except Exception as e:
                    continue

    print(f"Multi-TE LS fitting completed: {successful_fits}/{processed_voxels} successful fits "
          f"({100 * successful_fits / processed_voxels:.1f}%)")

    return att_map, cbf_map, rmse_map



def prepare_multite_data(pwi_meta):
    """
	Prepare Multi-TE data from metadata

	Parameters:
	- pwi_meta: metadata dictionary containing EchoTime, PostLabelingDelay, LabelingDuration

	Returns:
	- tis: unique inversion times [s]
	- tes: all echo times [s]
	- ntes: number of TEs per TI
	- taus: labeling durations [s]
	"""

    echo_times = np.array(pwi_meta["EchoTime"])
    plds = np.array(pwi_meta["PostLabelingDelay"])
    label_durations = np.array(pwi_meta["LabelingDuration"])

    # Get unique TIs (PLDs)
    tis_unique = np.unique(plds)

    # For each TI, find corresponding TEs
    tes_all = []
    ntes = []
    taus = []

    for ti in tis_unique:
        # Find indices for this TI
        ti_indices = np.where(np.isclose(plds, ti))[0]

        # Get TEs for this TI
        tes_for_ti = echo_times[ti_indices]
        tes_all.extend(tes_for_ti)

        # Number of TEs for this TI
        ntes.append(len(tes_for_ti))

        # Labeling duration for this TI (should be same for all TEs at this TI)
        tau_for_ti = label_durations[ti_indices[0]]
        taus.append(tau_for_ti)

    tes_array = np.array(tes_all)
    ntes_array = np.array(ntes)
    taus_array = np.array(taus)

    print(f"Multi-TE data preparation:")
    print(f"TIs: {tis_unique}")
    print(f"Number of TEs per TI: {ntes_array}")
    print(f"Total measurements: {len(tes_array)}")
    print(f"Labeling durations: {taus_array}")

    return tis_unique, tes_array, ntes_array, taus_array


