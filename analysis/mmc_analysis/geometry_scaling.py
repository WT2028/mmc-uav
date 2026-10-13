"""Fixed-body-mass trade-off and body-size scaling calculations."""

import numpy as np

from pathlib import Path

import csv

MB = 1.25

M0 = 1.45

M_SLIDER0 = 0.05

MU0 = M_SLIDER0 / M0

L0 = 0.40

S_LIM0 = 0.15

G = 9.81

IB_XY0 = 0.0585

IB_Z0 = 0.0600

IM_XY0 = 0.0006

IM_Z0 = 0.0012

OMEGA_S = 20.0

ZETA_S = 1.20

def mass_from_mu(mu: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return single-slider mass m and total mass M for fixed body mass MB."""
    m = np.asarray(mu) * MB / (1.0 - 4.0 * np.asarray(mu))
    M = MB + 4.0 * m
    return m, M

def inertia_matrix(mu: float, l: float = L0, chi: float = 0.0, upsilon: float = 0.0,
                   *, scale_body_with_l: bool = False) -> np.ndarray:
    """Compute I_s^1 from the parallel-axis relation for given parameters."""
    m, _ = mass_from_mu(float(mu))
    l_scale2 = (l / L0) ** 2 if scale_body_with_l else 1.0
    m_scale = m / M_SLIDER0
    ib = np.diag([IB_XY0 * l_scale2, IB_XY0 * l_scale2, IB_Z0 * l_scale2])
    im = np.diag([IM_XY0 * m_scale, IM_XY0 * m_scale, IM_Z0 * m_scale])
    rc = np.array([2.0 * mu * chi, 2.0 * mu * upsilon, 0.0])
    positions = [
        np.array([chi, 0.5 * l, 0.0]),
        np.array([0.5 * l, upsilon, 0.0]),
        np.array([chi, -0.5 * l, 0.0]),
        np.array([-0.5 * l, upsilon, 0.0]),
    ]
    eye = np.eye(3)
    rb = -rc
    I = ib + MB * ((rb @ rb) * eye - np.outer(rb, rb))
    for r0 in positions:
        r = r0 - rc
        I += im + m * ((r @ r) * eye - np.outer(r, r))
    return I

def effective_inertia(mu: float, l: float = L0, *, scale_body_with_l: bool = False) -> float:
    I = inertia_matrix(mu, l=l, scale_body_with_l=scale_body_with_l)
    return 0.5 * (I[0, 0] + I[1, 1])

def mass_ratio_table() -> None:
    """Compute authority, acceleration authority, and drive burden from formulas."""
    mu = np.linspace(0.020, 0.078, 360)
    m, M = mass_from_mu(mu)
    Ie = np.array([effective_inertia(float(v)) for v in mu])
    tau = 2.0 * mu * M * G * S_LIM0
    acc = tau / Ie
    drive = m * OMEGA_S**2 * S_LIM0

    _, M_base = mass_from_mu(MU0)
    Ie_base = effective_inertia(MU0)
    tau_base = 2.0 * MU0 * M_base * G * S_LIM0
    acc_base = tau_base / Ie_base
    drive_base = M_SLIDER0 * OMEGA_S**2 * S_LIM0

    tau_n = tau / tau_base
    acc_n = acc / acc_base
    drive_n = drive / drive_base

    write_csv(
        "mass_ratio_effectiveness_tradeoff",
        ["mu", "single_slider_mass_kg", "total_mass_kg", "I_e_kgm2", "tau_max_Nm",
         "attitude_accel_rad_s2", "drive_force_N", "tau_norm", "acc_norm", "drive_norm"],
        np.column_stack([mu, m, M, Ie, tau, acc, drive, tau_n, acc_n, drive_n]),
    )



def body_size_table() -> None:
    l = np.linspace(0.24, 0.76, 360)
    lam = S_LIM0 / L0
    _, M_base = mass_from_mu(MU0)
    Ie = np.array([effective_inertia(MU0, l=float(v), scale_body_with_l=True) for v in l])
    fixed = 2.0 * MU0 * M_base * G * S_LIM0 / Ie
    scaled = 2.0 * MU0 * M_base * G * (lam * l) / Ie
    baseline = 2.0 * MU0 * M_base * G * S_LIM0 / effective_inertia(MU0, l=L0, scale_body_with_l=True)
    fixed_n = fixed / baseline
    scaled_n = scaled / baseline

    write_csv(
        "body_size_authority",
        ["body_size_l_m", "I_e_kgm2", "authority_fixed_stroke_norm", "authority_geometric_scaling_norm"],
        np.column_stack([l, Ie, fixed_n, scaled_n]),
    )



def write_csv(stem,headers,rows):
    out=Path(__file__).resolve().parents[2]/'results/geometry_scaling'
    out.mkdir(parents=True,exist_ok=True)
    with (out/(stem+'.csv')).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)

if __name__=='__main__':
    mass_ratio_table()
    body_size_table()
