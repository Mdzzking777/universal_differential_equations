"""
AFM DMT-KV Parameter Recovery using Multiple Shooting
Inspired by hudson_bay.jl from the Universal Differential Equations repository

Goal: Recover (ks, cs, Estar) and predict y(t) from x(t) observations only
"""

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using LinearAlgebra, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Plots
gr()
using JLD2, FileIO
using Statistics
using DelimitedFiles
using Random
using Printf

Random.seed!(1234)

# Create a name for saving
svname = "AFM_Scenario"

println("="^60)
println("AFM DMT-KV Parameter Recovery")
println("="^60)

## ============================================================================
## Part 1: Data Loading and Preprocessing
## ============================================================================

println("\n[1/7] Loading data from trajectory.csv...")

# Check if data exists
data_path = joinpath(dirname(@__DIR__), "trajectory.csv")
if !isfile(data_path)
    error("trajectory.csv not found! Please run generate_trajectory_DMT_KV.py first.")
end

# Load data
data = readdlm(data_path, ',', Float64, '\n'; skipstart=1)

t_data = data[:, 1]           # Time (s)
x_data = data[:, 2]           # Tip displacement (m) - Observable
y_data = data[:, 3]           # Sample motion (m) - NOT observable (ground truth for validation)
s_data = data[:, 4]           # Separation (m)
contact_status = Int.(data[:, 5])  # 0=non-contact, 1=contact

println("  Data points: $(length(t_data))")
println("  Time span: $(t_data[1]) to $(t_data[end]) s")
println("  Contact fraction: $(@sprintf("%.2f", 100*sum(contact_status)/length(contact_status)))%")

# Compute xdot (velocity) via numerical differentiation
xdot_data = diff(x_data) ./ diff(t_data)
push!(xdot_data, xdot_data[end])  # Pad to same length

## ============================================================================
## Part 2: Identify Contact Segments
## ============================================================================

println("\n[2/7] Identifying contact segments...")

function find_contact_segments(contact_status, t_data)
    segments = []
    in_contact = false
    start_idx = 0

    for i in 1:length(contact_status)
        if contact_status[i] == 1 && !in_contact
            # Enter contact
            start_idx = i
            in_contact = true
        elseif contact_status[i] == 0 && in_contact
            # Leave contact
            push!(segments, (start=start_idx, stop=i-1,
                           t_start=t_data[start_idx], t_stop=t_data[i-1]))
            in_contact = false
        end
    end

    # Handle last segment
    if in_contact
        push!(segments, (start=start_idx, stop=length(contact_status),
                       t_start=t_data[start_idx], t_stop=t_data[end]))
    end

    return segments
end

contact_segments = find_contact_segments(contact_status, t_data)
N_segments = length(contact_segments)

println("  Found $(N_segments) contact segments")
if N_segments > 0
    seg_lengths = [seg.stop - seg.start + 1 for seg in contact_segments]
    println("  Segment lengths: min=$(minimum(seg_lengths)), max=$(maximum(seg_lengths)), mean=$(@sprintf("%.1f", mean(seg_lengths)))")
end

if N_segments == 0
    error("No contact segments found! Check trajectory data.")
end

## ============================================================================
## Part 3: Define AFM Parameters and Dynamics
## ============================================================================

println("\n[3/7] Setting up AFM model...")

# Fixed parameters (known from generate_trajectory_DMT_KV.py)
const k = 29.9
const f0 = 313.57e3
const wd = 2.0 * π * f0
const Q = 371.0
const m = k / (wd^2)
const c = m * wd / Q
const Fd = 2.05e-9
const Fadh = 2.0e-9
const R = 10e-9
const dist = 24e-9

# Ground truth (for validation only - pretend we don't know these)
const ks_true = 0.1
const cs_true = 0.24e-6
const Estar_true = 15e6

println("  Fixed parameters:")
println("    m = $(@sprintf("%.3e", m)) kg")
println("    c = $(@sprintf("%.3e", c)) kg/s")
println("    k = $(k) N/m")
println("    wd = $(@sprintf("%.3e", wd)) rad/s")
println("    Fd = $(@sprintf("%.3e", Fd)) N")
println("    Fadh = $(@sprintf("%.3e", Fadh)) N")
println("    R = $(@sprintf("%.3e", R)) m")
println("    dist = $(@sprintf("%.3e", dist)) m")

println("  Ground truth (unknown in real scenario):")
println("    ks = $(ks_true) N/m")
println("    cs = $(@sprintf("%.3e", cs_true)) kg/s")
println("    Estar = $(@sprintf("%.3e", Estar_true)) Pa")

# AFM Contact Dynamics (following the python code structure)
function afm_contact!(du, u, p, t)
    x, xdot, y = u
    ks, cs, Estar = p.ks, p.cs, p.Estar

    # Separation distance
    s = dist + x - y

    # Contact dynamics (we're in contact region)
    # Following generate_trajectory_DMT_KV.py line 66
    # Note: The python code uses y^1.5, which might be delta
    # We'll use the more standard formulation
    if y > 0
        F_hertz = (4.0/3.0) * Estar * sqrt(R) * (y^1.5)
    else
        F_hertz = 0.0
    end

    # Tip dynamics
    du[1] = xdot
    du[2] = (Fd * cos(wd * t) - k*x - c*xdot + Fadh - F_hertz) / m

    # Sample dynamics (Kelvin-Voigt)
    du[3] = (Fadh - F_hertz - ks*y) / cs
end

## ============================================================================
## Part 4: Multiple Shooting Loss Function
## ============================================================================

println("\n[4/7] Defining Multiple Shooting loss function...")

# Predict function for a single segment
function predict_segment(θ, seg, t_data, x_data, xdot_data)
    seg_idx = seg.start:seg.stop
    t_seg = t_data[seg_idx]

    # Get segment index
    seg_num = findfirst(s -> s.start == seg.start, contact_segments)

    # Initial condition for this segment
    u0 = [
        x_data[seg.start],          # x: known
        xdot_data[seg.start],       # xdot: known
        θ.y0_segments[seg_num]      # y: unknown (to be optimized)
    ]

    # Time span
    tspan = (t_seg[1], t_seg[end])

    # Parameters
    p = (ks=θ.ks, cs=θ.cs, Estar=θ.Estar)

    # Solve ODE
    prob = ODEProblem(afm_contact!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=t_seg,
               abstol=1e-8, reltol=1e-8)

    return sol
end

# Multiple Shooting Loss (inspired by hudson_bay.jl)
function multiple_shooting_loss(θ)
    total_loss = 0.0

    # Loss 1: Match x data in each contact segment
    for (i, seg) in enumerate(contact_segments)
        sol = predict_segment(θ, seg, t_data, x_data, xdot_data)

        # Check if solve succeeded
        if sol.retcode != :Success
            return 1e10
        end

        # Match tip displacement x
        seg_idx = seg.start:seg.stop
        x_pred = sol[1, :]
        x_obs = x_data[seg_idx]

        segment_loss = sum(abs2, x_obs .- x_pred)
        total_loss += segment_loss
    end

    # Loss 2: Continuity constraint between segments
    # y evolves according to dy/dt = -ks*y/cs during non-contact
    continuity_weight = 1e3

    for i in 1:(N_segments-1)
        # End of current segment
        sol_current = predict_segment(θ, contact_segments[i], t_data, x_data, xdot_data)
        y_end = sol_current[3, end]

        # Time gap to next segment
        gap_time = contact_segments[i+1].t_start - contact_segments[i].t_stop

        # Analytical solution during non-contact: y(t) = y0 * exp(-ks/cs * t)
        y_next_predicted = y_end * exp(-θ.ks/θ.cs * gap_time)
        y_next_actual = θ.y0_segments[i+1]

        continuity_loss = abs2(y_next_actual - y_next_predicted)
        total_loss += continuity_weight * continuity_loss
    end

    # Loss 3: Initial condition constraint
    # First segment should start near y(0) = 0
    initial_weight = 1e4
    initial_loss = abs2(θ.y0_segments[1] - 0.0)
    total_loss += initial_weight * initial_loss

    # Loss 4: Parameter regularization (keep in physical range)
    reg_weight = 1e-6
    reg_loss = 0.0

    if θ.ks < 0.0 || θ.ks > 10.0
        reg_loss += abs2(θ.ks - 0.5)
    end
    if θ.cs < 1e-8 || θ.cs > 1e-3
        reg_loss += abs2(θ.cs - 1e-6)
    end
    if θ.Estar < 1e5 || θ.Estar > 1e9
        reg_loss += abs2(θ.Estar - 1e7)
    end

    total_loss += reg_weight * reg_loss

    return total_loss
end

## ============================================================================
## Part 5: Setup Optimization Problem
## ============================================================================

println("\n[5/7] Setting up optimization...")

# Initial guess for parameters
θ_initial = ComponentVector(
    # Global parameters (shared across all segments)
    ks = 0.15,           # Initial guess (true: 0.1)
    cs = 0.3e-6,         # Initial guess (true: 0.24e-6)
    Estar = 20e6,        # Initial guess (true: 15e6)

    # Local variables (y initial value for each contact segment)
    y0_segments = zeros(N_segments)
)

println("  Optimization variables: $(length(θ_initial))")
println("    Global parameters: 3 (ks, cs, Estar)")
println("    Local variables: $(N_segments) (y0 for each segment)")
println("  Initial guess:")
println("    ks = $(θ_initial.ks) (true: $(ks_true))")
println("    cs = $(@sprintf("%.3e", θ_initial.cs)) (true: $(@sprintf("%.3e", cs_true)))")
println("    Estar = $(@sprintf("%.3e", θ_initial.Estar)) (true: $(@sprintf("%.3e", Estar_true)))")

# Test initial loss
try
    L0 = multiple_shooting_loss(θ_initial)
    println("  Initial loss: $(@sprintf("%.6e", L0))")
catch e
    println("  Warning: Initial loss computation failed: $e")
end

## ============================================================================
## Part 6: Two-Stage Optimization (ADAM + BFGS)
## ============================================================================

println("\n[6/7] Starting optimization...")
println("  Strategy: Two-stage (ADAM → BFGS)")

# Callback
losses = Float64[]
callback = function (θ, l)
    push!(losses, l)
    if length(losses) % 20 == 0
        println("  Iter $(length(losses)): loss = $(@sprintf("%.6e", l))")
        println("    ks = $(@sprintf("%.4f", θ.u.ks)) (true: $(ks_true))")
        println("    cs = $(@sprintf("%.3e", θ.u.cs)) (true: $(@sprintf("%.3e", cs_true)))")
        println("    Estar = $(@sprintf("%.3e", θ.u.Estar)) (true: $(@sprintf("%.3e", Estar_true)))")
    end
    return false
end

# Optimization function
optf = OptimizationFunction((x, p) -> multiple_shooting_loss(x),
                            Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optf, θ_initial)

# Stage 1: ADAM (fast exploration)
println("\n  [Stage 1/2] ADAM optimization...")
res1 = solve(optprob, ADAM(0.01), callback=callback, maxiters=100)
println("  Stage 1 complete. Loss: $(@sprintf("%.6e", losses[end]))")

# Stage 2: BFGS (precise convergence)
println("\n  [Stage 2/2] BFGS optimization...")
optprob2 = remake(optprob, u0=res1.u)
res2 = solve(optprob2, BFGS(), callback=callback, maxiters=500)
println("  Stage 2 complete. Loss: $(@sprintf("%.6e", losses[end]))")

θ_optimal = res2.u

## ============================================================================
## Part 7: Results and Visualization
## ============================================================================

println("\n[7/7] Generating results and visualizations...")

# Print final results
println("\n" * "="^60)
println("FINAL RESULTS")
println("="^60)
println("Parameter Recovery:")
println("  ks:")
println("    True:      $(ks_true)")
println("    Recovered: $(@sprintf("%.6f", θ_optimal.ks))")
println("    Error:     $(@sprintf("%.2f", abs(θ_optimal.ks - ks_true)/ks_true * 100))%")
println("  cs:")
println("    True:      $(@sprintf("%.6e", cs_true))")
println("    Recovered: $(@sprintf("%.6e", θ_optimal.cs))")
println("    Error:     $(@sprintf("%.2f", abs(θ_optimal.cs - cs_true)/cs_true * 100))%")
println("  Estar:")
println("    True:      $(@sprintf("%.6e", Estar_true))")
println("    Recovered: $(@sprintf("%.6e", θ_optimal.Estar))")
println("    Error:     $(@sprintf("%.2f", abs(θ_optimal.Estar - Estar_true)/Estar_true * 100))%")
println("="^60)

# Reconstruct full trajectory with optimal parameters
println("\nReconstructing full trajectory...")

x_reconstructed = copy(x_data)
y_reconstructed = zeros(length(t_data))

for (i, seg) in enumerate(contact_segments)
    sol = predict_segment(θ_optimal, seg, t_data, x_data, xdot_data)
    seg_idx = seg.start:seg.stop

    x_reconstructed[seg_idx] = sol[1, :]
    y_reconstructed[seg_idx] = sol[3, :]
end

# Calculate errors
contact_mask = contact_status .== 1
x_error = norm(x_data[contact_mask] .- x_reconstructed[contact_mask]) / norm(x_data[contact_mask])
y_error = norm(y_data[contact_mask] .- y_reconstructed[contact_mask]) / norm(y_data[contact_mask])

println("Reconstruction errors (contact region only):")
println("  x (observable): $(@sprintf("%.2f", x_error * 100))%")
println("  y (hidden):     $(@sprintf("%.2f", y_error * 100))%")

## Visualization

println("\nCreating plots...")

# Plot 1: Loss history
p1 = plot(1:length(losses), losses,
         yaxis=:log10, xlabel="Iteration", ylabel="Loss",
         title="Optimization History",
         label="Loss", lw=2, color=:blue,
         legend=:topright)
vline!([100], label="ADAM→BFGS", color=:red, linestyle=:dash)
savefig(p1, joinpath(pwd(), "plots", "$(svname)_losses.pdf"))

# Plot 2: Tip displacement (x) comparison - Full range
p2 = plot(t_data .* 1e3, x_data .* 1e9,
         label="Observed", color=:black, alpha=0.6,
         xlabel="Time (ms)", ylabel="Tip Displacement (nm)",
         title="Tip Displacement Recovery (Full Range)")
plot!(p2, t_data .* 1e3, x_reconstructed .* 1e9,
      label="Reconstructed", color=:red, linestyle=:dash, lw=2)
# Shade contact regions
for seg in contact_segments
    vspan!([t_data[seg.start]*1e3, t_data[seg.stop]*1e3],
           alpha=0.1, color=:green, label=nothing)
end
savefig(p2, joinpath(pwd(), "plots", "$(svname)_tip_displacement.pdf"))

# Plot 2b: Tip displacement (x) - Zoomed detail (1.4000-1.4200 ms)
t_zoom_start = 1.4000e-3  # 1.4000 ms
t_zoom_end = 1.4200e-3    # 1.4200 ms
zoom_mask = (t_data .>= t_zoom_start) .& (t_data .<= t_zoom_end)

p2b = plot(t_data[zoom_mask] .* 1e3, x_data[zoom_mask] .* 1e9,
          label="Observed", color=:black, alpha=0.8, lw=2,
          xlabel="Time (ms)", ylabel="Tip Displacement (nm)",
          title="Tip Displacement - Detail (1.4000-1.4200 ms)",
          marker=:circle, markersize=3)
plot!(p2b, t_data[zoom_mask] .* 1e3, x_reconstructed[zoom_mask] .* 1e9,
      label="Reconstructed", color=:red, linestyle=:dash, lw=2,
      marker=:square, markersize=3)
# Shade contact regions in zoom
for seg in contact_segments
    seg_t_start = t_data[seg.start]
    seg_t_end = t_data[seg.stop]
    if seg_t_end >= t_zoom_start && seg_t_start <= t_zoom_end
        vspan!([max(seg_t_start, t_zoom_start)*1e3, min(seg_t_end, t_zoom_end)*1e3],
               alpha=0.2, color=:green, label=nothing)
    end
end
savefig(p2b, joinpath(pwd(), "plots", "$(svname)_tip_displacement_zoom.pdf"))

# Plot 3: Sample motion (y) prediction vs ground truth - Full range
p3 = plot(t_data .* 1e3, y_data .* 1e9,
         label="Ground Truth", color=:black, alpha=0.6,
         xlabel="Time (ms)", ylabel="Sample Motion (nm)",
         title="Sample Motion Prediction (Full Range)")
plot!(p3, t_data[contact_mask] .* 1e3, y_reconstructed[contact_mask] .* 1e9,
      label="Predicted", color=:blue, marker=:circle, markersize=2, linestyle=:dash)
savefig(p3, joinpath(pwd(), "plots", "$(svname)_sample_motion.pdf"))

# Plot 3b: Sample motion (y) - Zoomed detail (1.4000-1.4200 ms)
zoom_mask_contact = zoom_mask .& contact_mask

p3b = plot(t_data[zoom_mask] .* 1e3, y_data[zoom_mask] .* 1e9,
          label="Ground Truth", color=:black, alpha=0.8, lw=2,
          xlabel="Time (ms)", ylabel="Sample Motion (nm)",
          title="Sample Motion - Detail (1.4000-1.4200 ms)",
          marker=:circle, markersize=3)
plot!(p3b, t_data[zoom_mask_contact] .* 1e3, y_reconstructed[zoom_mask_contact] .* 1e9,
      label="Predicted", color=:blue, lw=2, linestyle=:dash,
      marker=:square, markersize=3)
# Shade contact regions in zoom
for seg in contact_segments
    seg_t_start = t_data[seg.start]
    seg_t_end = t_data[seg.stop]
    if seg_t_end >= t_zoom_start && seg_t_start <= t_zoom_end
        vspan!([max(seg_t_start, t_zoom_start)*1e3, min(seg_t_end, t_zoom_end)*1e3],
               alpha=0.2, color=:green, label=nothing)
    end
end
savefig(p3b, joinpath(pwd(), "plots", "$(svname)_sample_motion_zoom.pdf"))

# Plot 4: Combined overview (2x2 grid)
layout = @layout [a b; c d]
p_combined = plot(p1, p2b, p3b, p2, layout=layout, size=(1400, 1000))
savefig(p_combined, joinpath(pwd(), "plots", "$(svname)_overview.pdf"))

println("  Plots saved to ./plots/")

## Save results
println("\nSaving results...")

save(joinpath(pwd(), "results", "$(svname)_results.jld2"),
    "t_data", t_data,
    "x_data", x_data,
    "y_data", y_data,
    "x_reconstructed", x_reconstructed,
    "y_reconstructed", y_reconstructed,
    "contact_segments", contact_segments,
    "theta_initial", θ_initial,
    "theta_optimal", θ_optimal,
    "losses", losses,
    "ks_true", ks_true,
    "cs_true", cs_true,
    "Estar_true", Estar_true
)

println("  Results saved to ./results/$(svname)_results.jld2")

println("\n" * "="^60)
println("COMPLETE!")
println("="^60)
println("\nSummary:")
println("  - Optimized $(length(θ_optimal)) variables (3 global + $(N_segments) local)")
println("  - Used $(length(contact_segments)) contact segments")
println("  - Final loss: $(@sprintf("%.6e", losses[end]))")
println("  - Parameter errors: ks $(@sprintf("%.1f", abs(θ_optimal.ks-ks_true)/ks_true*100))%, " *
        "cs $(@sprintf("%.1f", abs(θ_optimal.cs-cs_true)/cs_true*100))%, " *
        "Estar $(@sprintf("%.1f", abs(θ_optimal.Estar-Estar_true)/Estar_true*100))%")
println("  - Reconstruction error: x $(@sprintf("%.2f", x_error*100))%, y $(@sprintf("%.2f", y_error*100))%")
println("\nFiles generated:")
println("  - plots/$(svname)_*.pdf (6 plots)")
println("  - results/$(svname)_results.jld2")
println("="^60)
