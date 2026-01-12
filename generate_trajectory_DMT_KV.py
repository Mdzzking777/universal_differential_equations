"""
Generate DMT-KV Trajectory
Fixed parameters, minimal code
"""

import numpy as np

# ============================================================================
# Fixed Parameters
# ============================================================================

# Cantilever
k = 29.9
f0 = 313.57e3
wd = 2.0 * np.pi * f0
Q = 371.0
m = k / (wd**2)
c = m * wd / Q

# Contact & geometry
Estar = 15e6
R = 10e-9
Fad = 2.0e-9
dist = 24e-9

# Drive
Fd = 2.05e-9

# Surface (Kelvin-Voigt)
ks = 0.1
cs = 0.24e-6

# Simulation
t_end = 2e-3
nsteps = 125000
dt = t_end / nsteps

# ============================================================================
# RK4 Solver
# ============================================================================

def rk4_step(f, t, X, dt):
    k1 = f(t, X)
    k2 = f(t + 0.5*dt, X + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, X + 0.5*dt*k2)
    k4 = f(t + dt, X + dt*k3)
    return X + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# DMT-KV Model
# ============================================================================

def rhs_dmt_kv(t, X):
    """
    B) MOVING SURFACE (Kelvin-Voigt) + Hertz-DMT

    State: X = [x, v, y]
    """
    x, v, y = X

    # Separation
    s = dist + x - y

    # Contact detection
    if s <= 0:  # Contact
        F_Hertz = (4.0/3.0) * Estar * np.sqrt(R) * (y ** 1.5) if y > 0 else 0.0
        dvdt = (Fd * np.cos(wd * t) - k*x - c*v + Fad - F_Hertz) / m
        dydt = (Fad - F_Hertz - ks*y) / cs
    else:  # Non-contact
        dvdt = (Fd * np.cos(wd * t) - k*x - c*v) / m
        dydt = -ks * y / cs

    dxdt = v

    return np.array([dxdt, dvdt, dydt])

# ============================================================================
# Simulation
# ============================================================================

print("Simulating...")

# Time array
t = np.linspace(0, t_end, nsteps + 1)

# State arrays
x = np.zeros(nsteps + 1)
v = np.zeros(nsteps + 1)
y = np.zeros(nsteps + 1)

# Initial condition
X = np.array([0.0, 0.0, 0.0])
x[0], v[0], y[0] = X

# RK4 integration
for i in range(nsteps):
    X = rk4_step(rhs_dmt_kv, t[i], X, dt)
    x[i+1], v[i+1], y[i+1] = X

# Compute separation and contact
s = dist + x - y
contact = (s <= 0)

print(f"Complete: {len(t)} time points")
print(f"Contact fraction: {contact.sum()/len(contact)*100:.2f}%")

# ============================================================================
# Save Data
# ============================================================================

# Save as CSV
data = np.column_stack([t, x, y, s, contact.astype(int)])
np.savetxt('trajectory.csv', data, delimiter=',',
           header='time_s,x_tip_m,y_sample_m,s_separation_m,contact_status',
           comments='')

print("Saved: trajectory.csv")
