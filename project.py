import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from astropy.time import Time
from astropy.coordinates import SkyCoord, Galactic, ICRS
import astropy.units as u

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
def load_fermi_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.lower()
    rename_map = {}
    if 'mjd400d' in df.columns:
        rename_map['mjd400d'] = 'mjd'
    if 'decl' in df.columns:
        rename_map['decl'] = 'dec'
    df.rename(columns=rename_map, inplace=True)
    return df

# ------------------------------------------------------------
# 2. Координатные преобразования
# ------------------------------------------------------------
def gal_to_eq(l, b):
    gal = Galactic(l=l*u.deg, b=b*u.deg)
    eq = gal.transform_to(ICRS())
    return eq.ra.deg, eq.dec.deg

def filter_by_radius(df, center_ra, center_dec, radius_deg):
    if df.empty:
        return pd.DataFrame()
    center = SkyCoord(ra=center_ra*u.deg, dec=center_dec*u.deg)
    coords = SkyCoord(ra=df['ra'].values*u.deg, dec=df['dec'].values*u.deg)
    sep = center.separation(coords)
    return df[sep < radius_deg*u.deg].copy()

# ------------------------------------------------------------
# 3. Параметры триггера
# ------------------------------------------------------------
l_grb, b_grb = 51.5, 26.9
t_met = 273582308.313
radius = 1.0
target_window_days = 0.005
window_exclusion = 0.05

fermi_epoch = Time('2001-01-01T00:00:00', format='isot', scale='utc')
t0_mjd = fermi_epoch.mjd + (t_met / 86400.0)
ra_c, dec_c = gal_to_eq(l_grb, b_grb)

# ------------------------------------------------------------
# 4. Загрузка или генерация данных
# ------------------------------------------------------------
try:
    df_fermi = load_fermi_data('fermi_test_grb.json')
except FileNotFoundError:
    np.random.seed(42)
    bg_rate_day = 800
    t_start, t_end = t0_mjd - 0.5, t0_mjd + 0.5
    n_bg = np.random.poisson(bg_rate_day * (t_end - t_start))
    mock_bg_times = np.random.uniform(t_start, t_end, n_bg)
    n_signal = 350
    mock_signal_times = np.random.exponential(0.002, n_signal) + (t0_mjd - 0.0008)
    mock_times = np.concatenate([mock_bg_times, mock_signal_times])
    mock_ra = np.random.normal(ra_c, 0.3, len(mock_times))
    mock_dec = np.random.normal(dec_c, 0.3, len(mock_times))
    df_fermi = pd.DataFrame({'mjd': mock_times, 'ra': mock_ra, 'dec': mock_dec})

df_region = filter_by_radius(df_fermi, ra_c, dec_c, radius)
if df_region.empty:
    print("Ошибка: нет фотонов.")
    exit()

mjds = df_region['mjd'].values

# ------------------------------------------------------------
# 5. Анализ данных
# ------------------------------------------------------------
base_bin_days = 0.001
bins = np.arange(mjds.min(), mjds.max() + base_bin_days, base_bin_days)
counts, _ = np.histogram(mjds, bins=bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_seconds = base_bin_days * 86400
rate_raw = counts / bin_seconds

bg_mask = (bin_centers < (t0_mjd - window_exclusion)) | (bin_centers > (t0_mjd + window_exclusion))
bg_counts = counts[bg_mask]
mu_bg = np.mean(bg_counts)

window_bins = 5
counts_window_sum = pd.Series(counts).rolling(window=window_bins, center=True, min_periods=1).sum().values
mu_window = mu_bg * window_bins

pre_trial_p_values = poisson.sf(counts_window_sum - 1, mu_window)
pre_trial_p_values = np.clip(pre_trial_p_values, 1e-320, 1.0)
pre_trial_significance = -np.log10(pre_trial_p_values)

target_mask = (bin_centers >= (t0_mjd - target_window_days)) & (bin_centers <= (t0_mjd + target_window_days))
n_total_bins = len(bins) - 1
n_eff_trials = max(1, int(np.ceil(n_total_bins / window_bins)))

post_trial_p_values = np.ones_like(pre_trial_p_values)
post_trial_p_values[target_mask] = 1.0 - (1.0 - pre_trial_p_values[target_mask]) ** n_eff_trials
post_trial_p_values = np.clip(post_trial_p_values, 1e-320, 1.0)
post_trial_significance = -np.log10(post_trial_p_values)

sigma5_p = norm.sf(5)
pre_trial_threshold = -np.log10(sigma5_p)
post_trial_threshold = -np.log10(0.05)

# ------------------------------------------------------------
# 6. Визуализация
# ------------------------------------------------------------
counts_smoothed = pd.Series(counts).rolling(window=window_bins, center=True, min_periods=1).mean().values
rate_smoothed = counts_smoothed / bin_seconds

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.step(bin_centers, rate_raw, where='mid', color='lightgray', alpha=0.7, label='Исходные данные')
ax1.plot(bin_centers, rate_smoothed, color='blue', linewidth=2, label='Скользящая сумма')
ax1.axvline(x=t0_mjd, color='darkred', linestyle='--', linewidth=1.2, label='Триггер T0')
ax1.set_ylabel('Темп счёта (ф/с)')
ax1.legend(loc='upper right', frameon=False)
ax1.grid(alpha=0.2)

ax2.step(bin_centers, pre_trial_significance, where='mid', color='teal', linewidth=1.5, label='Pre-trial значимость')
ax2.step(bin_centers, post_trial_significance, where='mid', color='orange', linewidth=1.2, linestyle='--', label='Post-trial значимость')
ax2.axvline(x=t0_mjd, color='darkred', linestyle='--', linewidth=1.2)
ax2.axvspan(t0_mjd - target_window_days, t0_mjd + target_window_days, color='gold', alpha=0.08, label='Окно поиска')
ax2.axhline(y=pre_trial_threshold, color='red', linestyle=':', linewidth=1.5, label=f'Порог 5σ ({pre_trial_threshold:.1f})')
ax2.axhline(y=post_trial_threshold, color='green', linestyle=':', linewidth=1.2, label=f'Post-trial порог ({post_trial_threshold:.1f})')
ax2.set_xlabel('MJD (дни)')
ax2.set_ylabel('Значимость [-log10(p)]')
ax2.legend(loc='upper right', frameon=False)
ax2.grid(alpha=0.2)

for ax in (ax1, ax2):
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.xaxis.get_major_formatter().set_scientific(False)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Результаты
# ------------------------------------------------------------
target_indices = np.where(target_mask)[0]
if len(target_indices) > 0:
    best_local_idx = np.argmax(pre_trial_significance[target_mask])
    best_idx = target_indices[best_local_idx]
    best_pre_p = pre_trial_p_values[best_idx]
    best_pre_sig = pre_trial_significance[best_idx]
    best_post_p = post_trial_p_values[best_idx]
    best_post_sig = post_trial_significance[best_idx]
    delta_t_sec = (bin_centers[best_idx] - t0_mjd) * 86400.0

    print(f"Пик: {bin_centers[best_idx]:.6f} MJD (Δt={delta_t_sec:.1f}с)")
    print(f"Pre-trial: {best_pre_sig:.2f}σ (p={best_pre_p:.2e})")
    print(f"Post-trial: {best_post_sig:.2f}σ (p={best_post_p:.2e})")
    print(f"Статус: {'ОБНАРУЖЕНО' if best_pre_sig >= pre_trial_threshold else 'НЕ ОБНАРУЖЕНО'}")