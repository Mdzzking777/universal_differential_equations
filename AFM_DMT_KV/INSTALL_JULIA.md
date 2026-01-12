# Julia Installation Guide for AFM Parameter Recovery

## Installation Steps

### Option 1: Official Julia Installation (Recommended)

1. **Download Julia**
   - Visit: https://julialang.org/downloads/
   - Download the latest stable version (1.10.x recommended)
   - For Windows: Choose "64-bit (installer)" for easiest setup

2. **Run the Installer**
   - Execute the downloaded .exe file
   - **IMPORTANT**: Check "Add Julia to PATH" during installation
   - Accept default installation location or choose your preferred directory

3. **Verify Installation**
   Open a new terminal and run:
   ```bash
   julia --version
   ```
   You should see something like: `julia version 1.10.x`

### Option 2: Using juliaup (Modern Package Manager)

1. **Install juliaup**
   Open PowerShell and run:
   ```powershell
   winget install julia -s msstore
   ```

2. **Verify Installation**
   ```bash
   julia --version
   ```

## After Julia is Installed

### Quick Start (Automated)

Once Julia is installed, simply run:

```bash
cd "c:\Users\Public\Documents\GitHub\universal_differential_equations\AFM_DMT_KV"
julia --project=. run_all.jl
```

This will:
- Check for trajectory.csv (already generated ✓)
- Install all required Julia packages automatically
- Run the parameter recovery optimization
- Generate plots and save results

**Expected runtime**: 5-10 minutes (first run includes package installation)

### Manual Step-by-Step

If you prefer to run manually:

1. **Set up the Julia environment**
   ```bash
   cd "c:\Users\Public\Documents\GitHub\universal_differential_equations\AFM_DMT_KV"
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

2. **Run the main scenario**
   ```bash
   julia --project=. scenario_afm.jl
   ```

3. **Analyze results** (optional)
   ```bash
   julia --project=. analyze_results.jl
   ```

## Expected Output

After successful execution, you'll find:

### Plots (./plots/ folder)
1. AFM_Scenario_losses.pdf - Optimization convergence
2. AFM_Scenario_tip_displacement.pdf - x(t) reconstruction
3. AFM_Scenario_sample_motion.pdf - **y(t) prediction vs ground truth**
4. AFM_Scenario_phase_space.pdf - Phase portrait
5. AFM_Scenario_overview.pdf - Combined overview
6-9. Additional diagnostic plots (if running analyze_results.jl)

### Data (./results/ folder)
- AFM_Scenario_results.jld2 - All results saved for later analysis

### Console Output
```
[6/7] Starting optimization...
  [Stage 1/2] ADAM optimization...
  Iter 20: loss = 1.234e-06
    ks = 0.1234 (true: 0.1)
    cs = 2.345e-07 (true: 2.400e-07)
    Estar = 1.456e+07 (true: 1.500e+07)

  [Stage 2/2] BFGS optimization...
  Final: ks error = 2.3%, cs error = 3.1%, Estar error = 4.5%
```

## Troubleshooting

### Issue: "julia: command not found" (even after installation)

**Solution**: Restart your terminal or add Julia to PATH manually

To find Julia installation:
```bash
# Windows - look in:
C:\Users\<YourUsername>\AppData\Local\Programs\Julia-1.x.x\bin\julia.exe
# Or
C:\Program Files\Julia-1.x.x\bin\julia.exe
```

Add to PATH:
1. Search "Environment Variables" in Windows
2. Edit "Path" variable
3. Add Julia's `bin` folder
4. Restart terminal

### Issue: Package installation errors

**Solution**:
```bash
julia --project=. -e "using Pkg; Pkg.update(); Pkg.instantiate()"
```

### Issue: Optimization not converging

**Possible causes**:
- Poor initial guess
- Insufficient contact segments
- Try adjusting parameters in scenario_afm.jl line 242

## Current Status

✓ trajectory.csv generated (125,001 points, 8.36% contact)
✓ All Julia code files ready
✓ Project.toml with dependencies configured
⏳ Waiting for Julia installation

## Next Steps After Installation

1. Open a NEW terminal window
2. Navigate to AFM_DMT_KV folder
3. Run: `julia --project=. run_all.jl`
4. Wait 5-10 minutes for optimization
5. Check plots/ folder for results
6. Compare recovered parameters with ground truth:
   - True: ks=0.1, cs=0.24e-6, Estar=15e6
   - Expected error: <5-10%

## Support

If you encounter any issues after installing Julia, please share:
- Julia version: `julia --version`
- Error message
- Which step failed
