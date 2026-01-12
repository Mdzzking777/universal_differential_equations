# AFM DMT-KV Parameter Recovery

Recovery of AFM (Atomic Force Microscopy) parameters using Universal Differential Equations and Multiple Shooting strategy.

## Problem Description

**Goal**: Recover unknown material parameters `(ks, cs, Estar)` and predict hidden state `y(t)` from partial observations.

**Known**:
- Tip displacement: `x(t)` (fully observed)
- Initial condition: `y(0) = 0`
- All other parameters: `m, c, k, Fadh, R, wd, dist, Fd`
- Complete equation structure (DMT-Hertz contact + Kelvin-Voigt surface)

**Unknown**:
- Surface stiffness: `ks`
- Surface damping: `cs`
- Effective elastic modulus: `Estar`
- Sample motion trajectory: `y(t)` for t > 0

## Method

Inspired by `hudson_bay.jl` from the UDE repository, uses:
- **Multiple Shooting**: Split contact regions into segments
- **Continuity Constraints**: Connect segments via analytical non-contact solution
- **Two-Stage Optimization**: ADAM (exploration) → BFGS (refinement)

## Usage

### Step 1: Generate Synthetic Data

First, run the Python script to generate training data:

```bash
cd ..
python generate_trajectory_DMT_KV.py
```

This creates `trajectory.csv` with ground truth data.

### Step 2: Run Parameter Recovery

```bash
cd AFM_DMT_KV
julia scenario_afm.jl
```

This will:
1. Load data from `trajectory.csv`
2. Identify contact segments
3. Optimize parameters using Multiple Shooting
4. Generate plots and save results

## Output

### Plots (in `./plots/`)

1. `AFM_Scenario_losses.pdf` - Optimization history
2. `AFM_Scenario_tip_displacement.pdf` - x(t) reconstruction
3. `AFM_Scenario_sample_motion.pdf` - y(t) prediction vs ground truth
4. `AFM_Scenario_phase_space.pdf` - Phase portrait (x vs y)
5. `AFM_Scenario_overview.pdf` - Combined overview

### Data (in `./results/`)

`AFM_Scenario_results.jld2` - Contains:
- Optimized parameters
- Reconstructed trajectories
- Loss history
- Contact segment information

## Key Features

### Multiple Shooting Strategy

Unlike simple global optimization, Multiple Shooting:
- Splits long trajectory into short segments (each contact event)
- Optimizes initial `y` for each segment independently
- Connects segments via physical continuity constraints
- Reduces error accumulation from unobserved state `y`

### Optimization Variables

```
θ = [ks, cs, Estar, y0₁, y0₂, ..., y0ₙ]
     └─ Global (3) ─┘  └─── Local (N) ────┘
```

- **Global**: Physical parameters shared across all segments
- **Local**: Initial sample position for each contact segment

Typical: 3 + 10 = 13 variables (manageable!)

## Algorithm Flow

```
1. Load x(t) data → Extract contact segments
                  ↓
2. Define loss:  Match x in each segment
                 + Continuity constraint
                 + Initial condition
                  ↓
3. Optimize:     ADAM (100 iter) → BFGS (500 iter)
                  ↓
4. Validate:     Compare with ground truth y(t)
                  ↓
5. Visualize:    6 diagnostic plots
```

## Physics

**Contact Dynamics** (when tip touches sample):
```
dx/dt = v
dv/dt = [F_drive - F_adhesion + F_Hertz - F_spring - F_damping] / m
dy/dt = [F_adhesion - F_Hertz - k_s*y] / c_s
```

**Non-Contact Dynamics** (analytical):
```
y(t) = y₀ * exp(-k_s/c_s * t)
```

**Hertz-DMT Contact Force**:
```
F_Hertz = (4/3) * E* * √R * δ^1.5
```

## References

- Rackauckas et al., "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385
- `hudson_bay.jl` - Multiple Shooting implementation for sparse data
- `scenario_2.jl` - Partial observation handling

## Notes

- The code assumes `trajectory.csv` exists in the parent directory
- Ground truth parameters: `ks=0.1`, `cs=0.24e-6`, `Estar=15e6`
- Initial guesses: `ks=0.15`, `cs=0.3e-6`, `Estar=20e6` (intentionally offset)
- Optimization typically converges in 200-600 iterations
- Expected parameter recovery accuracy: <10% error
