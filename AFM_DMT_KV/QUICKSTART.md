# Quick Start Guide

## Prerequisites

- Python 3.x with NumPy
- Julia 1.7+

## Step-by-Step Instructions

### 1. Generate Synthetic Data

```bash
cd ..
python generate_trajectory_DMT_KV.py
```

**Output**: `trajectory.csv` (125,001 data points)

**What it contains**:
- `time_s`: Time in seconds
- `x_tip_m`: Tip displacement (observable)
- `y_sample_m`: Sample motion (NOT observable - for validation only)
- `s_separation_m`: Separation distance
- `contact_status`: 0=non-contact, 1=contact

---

### 2. Run Parameter Recovery

```bash
cd AFM_DMT_KV
julia run_all.jl
```

**What it does**:
1. Checks for `trajectory.csv`
2. Sets up Julia environment (installs packages on first run)
3. Runs `scenario_afm.jl`

**Expected runtime**: 2-5 minutes (first run may take longer for package installation)

---

### 3. Analyze Results

```bash
julia analyze_results.jl
```

**Output**: Additional diagnostic plots and statistical analysis

---

## Expected Results

### Parameter Recovery

| Parameter | True Value | Expected Recovery | Typical Error |
|-----------|-----------|-------------------|---------------|
| ks | 0.1 N/m | ~0.095-0.105 | <5% |
| cs | 0.24 µN·s/m | ~0.23-0.25 | <5% |
| E* | 15 MPa | ~14-16 MPa | <10% |

### State Reconstruction

- **x (observable)**: <1% NRMSE
- **y (hidden)**: <10% NRMSE

---

## Output Files

### Plots (`./plots/`)

1. **AFM_Scenario_losses.pdf**
   - Optimization convergence

2. **AFM_Scenario_tip_displacement.pdf**
   - Observable x(t) reconstruction

3. **AFM_Scenario_sample_motion.pdf**
   - Hidden y(t) prediction vs ground truth

4. **AFM_Scenario_phase_space.pdf**
   - Phase portrait (x vs y)

5. **AFM_Scenario_overview.pdf**
   - Combined 4-panel overview

6. **AFM_Scenario_residuals.pdf** (from analyze_results.jl)
   - Residual analysis

7. **AFM_Scenario_scatter.pdf** (from analyze_results.jl)
   - Predicted vs true scatter plots

8. **AFM_Scenario_segments.pdf** (from analyze_results.jl)
   - Individual segment comparisons

9. **AFM_Scenario_parameters.pdf** (from analyze_results.jl)
   - Parameter comparison bar charts

### Data (`./results/`)

**AFM_Scenario_results.jld2** contains:
```julia
- t_data              # Time array
- x_data              # Observed tip displacement
- y_data              # Ground truth sample motion
- x_reconstructed     # Reconstructed tip displacement
- y_reconstructed     # Predicted sample motion
- contact_segments    # Contact segment information
- theta_initial       # Initial parameter guess
- theta_optimal       # Optimized parameters
- losses              # Loss history
- ks_true, cs_true, Estar_true  # Ground truth parameters
```

---

## Troubleshooting

### Error: "trajectory.csv not found"

**Solution**: Run the Python data generation script first:
```bash
cd ..
python generate_trajectory_DMT_KV.py
```

---

### Error: Package installation issues

**Solution**: Manually instantiate packages:
```bash
cd AFM_DMT_KV
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

### Poor parameter recovery

**Possible causes**:
1. Insufficient contact events (need >10)
2. Very noisy data
3. Poor initial guess (try adjusting in `scenario_afm.jl` line 242)

**Solution**: Check contact segments:
```bash
julia -e 'using DelimitedFiles; data = readdlm("../trajectory.csv", ",", Float64; skipstart=1); println("Contact fraction: ", sum(data[:,5])/length(data[:,5])*100, "%")'
```

Should be >5% for good recovery.

---

## Customization

### Modify initial guess (scenario_afm.jl, line 242)

```julia
θ_initial = ComponentVector(
    ks = 0.15,      # Try values in range [0.05, 0.5]
    cs = 0.3e-6,    # Try values in range [0.1e-6, 1e-6]
    Estar = 20e6,   # Try values in range [10e6, 50e6]
    y0_segments = zeros(N_segments)
)
```

### Adjust optimization settings (scenario_afm.jl, lines 285-291)

```julia
# Stage 1: ADAM
res1 = solve(optprob, ADAM(0.01), maxiters=100)  # Increase maxiters if needed

# Stage 2: BFGS
res2 = solve(optprob2, BFGS(), maxiters=500)     # Increase maxiters if needed
```

### Change loss weights (scenario_afm.jl, lines 215-230)

```julia
continuity_weight = 1e3    # Increase to enforce stronger continuity
initial_weight = 1e4       # Increase to enforce y(0)≈0 more strictly
reg_weight = 1e-6          # Increase to prevent extreme parameter values
```

---

## Understanding the Output

### Console Output Example

```
[6/7] Starting optimization...
  Strategy: Two-stage (ADAM → BFGS)

  [Stage 1/2] ADAM optimization...
  Iter 20: loss = 1.234567e-06
    ks = 0.1234 (true: 0.1)
    cs = 2.345e-07 (true: 2.400e-07)
    Estar = 1.456e+07 (true: 1.500e+07)
```

**What to look for**:
- Loss should decrease steadily
- Parameters should converge toward true values
- Final errors should be <10%

---

## Next Steps

After successful recovery:

1. **Experiment with different initial conditions**
   - Modify `generate_trajectory_DMT_KV.py` line 92
   - Test robustness of recovery

2. **Add noise to synthetic data**
   - Add Gaussian noise to x_data in Python script
   - Test algorithm with realistic measurement noise

3. **Try real experimental data**
   - Format: CSV with columns [time, x_tip, contact_status]
   - Adjust `scenario_afm.jl` to load your data format

4. **Extend to non-contact region**
   - Currently only uses contact segments
   - Could add non-contact dynamics for better y(t) tracking

---

## Citation

If using this code, please cite:

- Rackauckas et al., "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385
- Original UDE repository: https://github.com/SciML/UniversalDifferentialEquations.jl

---

## Support

For issues or questions:
1. Check the main README.md
2. Review `scenario_afm.jl` comments
3. Consult original UDE examples (especially `hudson_bay.jl`)
