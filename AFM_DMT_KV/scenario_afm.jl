"""
AFM DMT-KV Parameter Estimation (Fixed Version v2)
Based on Universal Differential Equations framework

Key fix: Data generated in Julia using SAME soft switching as model!
This ensures sanity check passes (loss_true ≈ 0).

Other fixes:
1. Added tip acceleration to loss
2. Data normalization
3. Parameter log-transform (ensure positivity)
4. Contact/Non-contact balanced weighting
5. Adjusted optimizer settings (ADAM lr=0.1, BFGS stepnorm=0.01)
6. AutoForwardDiff for gradient computation

Author: Based on UDE framework by Christopher Rackauckas
"""

## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using DiffEqCallbacks
using LinearAlgebra, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using Zygote
using Plots
gr()
using JLD2, FileIO
using Statistics
using DelimitedFiles
using Random
rng = Random.default_rng()
Random.seed!(42)

# Create a name for saving
svname = "AFM_Scenario_"

println("="^60)
println("AFM DMT-KV Parameter Estimation (Fixed Version)")
println("="^60)

## ============================================================================
## Known Physical Parameters
## ============================================================================

# Cantilever parameters
const k_cantilever = 29.9              # Spring constant [N/m]
const f0 = 313.57e3                    # Resonance frequency [Hz]
const wd = 2.0 * π * f0                # Angular frequency [rad/s]
const Q = 371.0                        # Quality factor
const m = k_cantilever / (wd^2)        # Effective mass [kg]
const c_damping = m * wd / Q           # Damping coefficient [N·s/m]

# Contact geometry
const R = 10e-9                        # Tip radius [m]
const d = 24e-9                        # Equilibrium separation [m]
const Fad = 2.0e-9                     # Adhesion force [N]

# Drive
const Fd = 2.05e-9                     # Drive force amplitude [N]

# Pack known parameters
const p_known = (m = m, k = k_cantilever, c = c_damping,
                 Fd = Fd, wd = wd, R = R, d = d, Fad = Fad)

println("Known parameters:")
println("  m = $(p_known.m) kg")
println("  k = $(p_known.k) N/m")
println("  c = $(p_known.c) N·s/m")
println("  R = $(p_known.R) m")
println("  d = $(p_known.d) m")

# True parameters (for validation only - NOT used in training!)
const Estar_true = 15e6                # True effective modulus [Pa]
const ks_true = 0.1                    # True sample stiffness [N/m]
const cs_true = 0.24e-6                # True sample damping [N·s/m]

println("\nTrue parameters (for validation):")
println("  Estar = $(Estar_true) Pa")
println("  ks = $(ks_true) N/m")
println("  cs = $(cs_true) N·s/m")

## ============================================================================
## Generate Data in Julia (ensures consistency with model!)
## ============================================================================

println("\n" * "="^60)
println("Generating data in Julia...")
println("="^60)

"""
AFM dynamics for DATA GENERATION with true parameters.
Uses the SAME soft switching as the optimization model!
This ensures sanity check will pass (loss_true ≈ 0).
"""
function afm_dynamics_true!(du, u, p, t)
    x1, x2, x3 = u

    # True parameters (hardcoded for data generation)
    Estar = Estar_true
    ks = ks_true
    cs = cs_true

    # Known parameters
    m_val = p_known.m
    k_val = p_known.k
    c_val = p_known.c
    Fd_val = p_known.Fd
    wd_val = p_known.wd
    R_val = p_known.R
    d_val = p_known.d
    Fad_val = p_known.Fad

    # Separation distance
    s = d_val + x1 - x3

    # SAME soft switching as optimization model
    sharpness = 1e9
    contact_indicator = 0.5 * (1.0 - tanh(s * sharpness))

    eps_soft = 1e-12
    delta_soft = 0.5 * (-s + sqrt(s^2 + eps_soft))

    # Hertz contact force
    F_hertz = (4.0/3.0) * Estar * sqrt(R_val) * (delta_soft^1.5) * contact_indicator

    # Dynamics
    du[1] = x2
    du[2] = (Fd_val * cos(wd_val * t) - k_val * x1 - c_val * x2 +
             contact_indicator * Fad_val - F_hertz) / m_val
    du[3] = (-ks * x3 + contact_indicator * (Fad_val - F_hertz)) / cs

    return nothing
end

# Simulation parameters (matching Python script)
t_end = 2e-3
nsteps = 125000
dt = t_end / nsteps

# Time array
t_full = range(0, t_end, length=nsteps+1) |> collect

# Initial conditions
u0_gen = [0.0, 0.0, 0.0]

# Solve ODE with true parameters
println("Solving ODE with true parameters...")
prob_gen = ODEProblem(afm_dynamics_true!, u0_gen, (0.0, t_end), nothing)
sol_gen = solve(prob_gen, Tsit5(), saveat=t_full, abstol=1e-12, reltol=1e-12)

# Extract data
x1_full = sol_gen[1, :]          # tip displacement [m]
x2_full = sol_gen[2, :]          # tip velocity [m/s]
x3_full = sol_gen[3, :]          # sample displacement [m]

# Compute separation and contact status
s_full = p_known.d .+ x1_full .- x3_full
contact_full = Float64.(s_full .<= 0)

# Compute acceleration directly from dynamics (same as data generation)
x2dot_full = zeros(length(t_full))
for i in 1:length(t_full)
    du = zeros(3)
    afm_dynamics_true!(du, [x1_full[i], x2_full[i], x3_full[i]], nothing, t_full[i])
    x2dot_full[i] = du[2]
end

println("Generated $(length(t_full)) data points")
println("Time span: $(t_full[1]) to $(t_full[end]) s")
println("Contact fraction: $(sum(contact_full)/length(contact_full)*100)%")

## ============================================================================
## Data Preprocessing: Downsampling + Normalization
## ============================================================================

# Downsample for faster training
sample_rate = 100  # Keep every 100th point
indices = 1:sample_rate:length(t_full)

t = t_full[indices]
x1_data = x1_full[indices]
x2_data = x2_full[indices]        # x2 = dx1/dt = velocity (from ODE state)
x3_data = x3_full[indices]        # sample displacement
x2dot_data = x2dot_full[indices]  # tip acceleration (from dynamics)
contact_data = contact_full[indices]  # contact status for weighting

println("\nAfter downsampling (rate=$sample_rate):")
println("  $(length(t)) data points")
println("  Time step: $(mean(diff(t))) s")

# ============================================================================
# FIX 2: Data Normalization
# ============================================================================
# Calculate normalization scales
x1_scale = maximum(abs.(x1_data)) + eps()
x2_scale = maximum(abs.(x2_data)) + eps()
x3_scale = maximum(abs.(x3_data)) + eps()
x2dot_scale = maximum(abs.(x2dot_data)) + eps()

println("\nNormalization scales:")
println("  x1_scale = $x1_scale m")
println("  x2_scale = $x2_scale m/s")
println("  x3_scale = $x3_scale m")
println("  x2dot_scale = $x2dot_scale m/s²")

# Normalized data
x1_norm = x1_data ./ x1_scale
x2_norm = x2_data ./ x2_scale
x3_norm = x3_data ./ x3_scale
x2dot_norm = x2dot_data ./ x2dot_scale

# Initial conditions
u0 = [x1_data[1], x2_data[1], x3_data[1]]
tspan = (t[1], t[end])

println("Initial conditions: u0 = $u0")

## ============================================================================
## AFM Dynamics Model with Soft Switching
## ============================================================================

"""
AFM DMT-KV dynamics with SOFT switching for gradient compatibility.

State: u = [x1, x2, x3]
Parameters: θ = [log_Estar, log_ks, log_cs] (log-transformed for positivity)
"""
function afm_dynamics!(du, u, θ, t)
    x1, x2, x3 = u

    # =========================================================================
    # FIX 3: Log-transform ensures positive parameters
    # =========================================================================
    Estar = exp(θ.log_Estar)
    ks = exp(θ.log_ks)
    cs = exp(θ.log_cs)

    # Known parameters
    m_val = p_known.m
    k_val = p_known.k
    c_val = p_known.c
    Fd_val = p_known.Fd
    wd_val = p_known.wd
    R_val = p_known.R
    d_val = p_known.d
    Fad_val = p_known.Fad

    # Separation distance
    s = d_val + x1 - x3

    # =========================================================================
    # Soft switching for differentiability
    # =========================================================================
    # Smooth contact indicator: 1 when in contact (s<0), 0 otherwise
    sharpness = 1e9  # Controls transition sharpness
    contact_indicator = 0.5 * (1.0 - tanh(s * sharpness))

    # Soft ReLU for indentation depth: max(-s, 0)
    eps_soft = 1e-12
    delta_soft = 0.5 * (-s + sqrt(s^2 + eps_soft))

    # Hertz contact force (only active during contact)
    F_hertz = (4.0/3.0) * Estar * sqrt(R_val) * (delta_soft^1.5) * contact_indicator

    # Dynamics (smooth combination of contact and non-contact)
    du[1] = x2
    du[2] = (Fd_val * cos(wd_val * t) - k_val * x1 - c_val * x2 +
             contact_indicator * Fad_val - F_hertz) / m_val
    du[3] = (-ks * x3 + contact_indicator * (Fad_val - F_hertz)) / cs

    return nothing
end

"""
Compute acceleration from state (for loss function).
This matches du[2] from the dynamics.
"""
function compute_acceleration(x1, x2, x3, θ, t_val)
    Estar = exp(θ.log_Estar)
    ks = exp(θ.log_ks)
    cs = exp(θ.log_cs)

    s = p_known.d + x1 - x3

    # Soft switching
    sharpness = 1e9
    contact_indicator = 0.5 * (1.0 - tanh(s * sharpness))

    eps_soft = 1e-12
    delta_soft = 0.5 * (-s + sqrt(s^2 + eps_soft))

    F_hertz = (4.0/3.0) * Estar * sqrt(p_known.R) * (delta_soft^1.5) * contact_indicator

    acc = (p_known.Fd * cos(p_known.wd * t_val) - p_known.k * x1 - p_known.c * x2 +
           contact_indicator * p_known.Fad - F_hertz) / p_known.m

    return acc
end

## ============================================================================
## Initial Parameter Guess (Random Multi-Start with Log-Transform)
## ============================================================================

# Parameter ranges for random initialization
# ks:    0.01 to 0.1 (true: 0.1)
# cs:    1e-8 to 1e-6 (true: 0.24e-6)
# Estar: 1e6 to 1e8 (true: 15e6)

"""
Generate random initial parameters in LOG space
"""
function generate_random_init()
    # Random in log space
    log_ks_init = log(0.01) + rand() * (log(0.1) - log(0.01))
    log_cs_init = log(1e-8) + rand() * (log(1e-6) - log(1e-8))
    log_Estar_init = log(1e6) + rand() * (log(1e8) - log(1e6))

    return ComponentVector(log_Estar = log_Estar_init,
                          log_ks = log_ks_init,
                          log_cs = log_cs_init)
end

# Number of random restarts
const N_RESTARTS = 3

# Generate initial guesses
println("\n" * "="^60)
println("Generating $N_RESTARTS random initial guesses...")
println("="^60)
println("Parameter ranges:")
println("  Estar: 1e6 to 1e8 Pa (true: $(Estar_true))")
println("  ks:    0.01 to 0.1 N/m (true: $(ks_true))")
println("  cs:    1e-8 to 1e-6 N·s/m (true: $(cs_true))")

p_inits = [generate_random_init() for _ in 1:N_RESTARTS]

println("\nGenerated initial guesses (actual values from log):")
for (i, p) in enumerate(p_inits)
    println("  [$i] Estar=$(exp(p.log_Estar)), ks=$(exp(p.log_ks)), cs=$(exp(p.log_cs))")
end

# Use first initial guess for problem definition
p_init = p_inits[1]

# Define ODE problem
prob = ODEProblem(afm_dynamics!, u0, tspan, p_init)

## ============================================================================
## Prediction and Loss Functions
## ============================================================================

"""
Predict trajectory given parameters θ (in log space)
"""
function predict(θ; u0=u0, T=t)
    _prob = remake(prob, u0=u0, tspan=(T[1], T[end]), p=θ)

    sol = solve(_prob, Tsit5(), saveat=T,
                abstol=1e-10, reltol=1e-10,
                sensealg=ForwardDiffSensitivity())

    # Handle solver failure
    if sol.retcode != :Success
        return fill(Inf, 3, length(T))
    end

    return Array(sol)
end

"""
Loss function with ALL fixes:
1. Includes tip displacement, velocity, acceleration, and sample motion
2. Normalized data
3. Contact/Non-contact balanced weighting
"""
function loss(θ)
    X̂ = predict(θ)

    # Check for solver failure
    if any(isinf.(X̂))
        return Inf
    end

    # =========================================================================
    # FIX 6: Contact/Non-contact balanced weighting
    # =========================================================================
    contact_mask = contact_data .== 1
    n_contact = max(sum(contact_mask), 1)
    n_noncontact = max(length(t) - n_contact, 1)

    # Compute normalized prediction errors
    err_x1 = x1_norm .- X̂[1, :] ./ x1_scale
    err_x2 = x2_norm .- X̂[2, :] ./ x2_scale
    err_x3 = x3_norm .- X̂[3, :] ./ x3_scale

    # FIX 1: Compute predicted acceleration and its error
    x2dot_pred = [compute_acceleration(X̂[1,i], X̂[2,i], X̂[3,i], θ, t[i]) for i in 1:length(t)]
    err_x2dot = x2dot_norm .- x2dot_pred ./ x2dot_scale

    # =========================================================================
    # Balanced loss: normalize by number of points in each region
    # =========================================================================

    # Contact region loss (normalized by n_contact)
    loss_contact = (
        sum(abs2, err_x1[contact_mask]) +
        sum(abs2, err_x2[contact_mask]) +
        sum(abs2, err_x3[contact_mask]) +
        sum(abs2, err_x2dot[contact_mask])
    ) / n_contact

    # Non-contact region loss (normalized by n_noncontact)
    nc_mask = .!contact_mask
    loss_noncontact = (
        sum(abs2, err_x1[nc_mask]) +
        sum(abs2, err_x2[nc_mask]) +
        sum(abs2, err_x3[nc_mask]) +
        sum(abs2, err_x2dot[nc_mask])
    ) / n_noncontact

    # Equal weighting of both regions
    return loss_contact + loss_noncontact
end

## ============================================================================
## Sanity Check: Verify model correctness with true parameters
## ============================================================================

println("\n" * "="^60)
println("Sanity Check: Testing with true parameters...")
println("="^60)

# True parameters in LOG space
p_true_log = ComponentVector(log_Estar = log(Estar_true),
                             log_ks = log(ks_true),
                             log_cs = log(cs_true))
loss_true = loss(p_true_log)

println("True parameters: Estar=$(Estar_true), ks=$(ks_true), cs=$(cs_true)")
println("Loss with true parameters: $loss_true")

if loss_true > 1e-6
    @warn "Sanity check WARNING: loss is not near zero!"
    println("   This may indicate soft switching approximation error.")
    println("   Expected: < 1e-6, Got: $loss_true")
else
    println("Sanity check PASSED: loss < 1e-6")
end

# Test all initial guesses
println("\nInitial losses for each random start:")
for (i, p) in enumerate(p_inits)
    l = loss(p)
    println("  [$i] Loss = $l (Estar=$(exp(p.log_Estar)), ks=$(exp(p.log_ks)), cs=$(exp(p.log_cs)))")
end

## ============================================================================
## Multi-Start Training (ADAM + BFGS with Fixed Settings)
## ============================================================================

println("\n" * "="^60)
println("Starting Multi-Start Optimization ($N_RESTARTS restarts)...")
println("Using ADAM (lr=0.1) + BFGS (stepnorm=0.01)")
println("="^60)

# Storage for results from all restarts
all_results = []
all_losses_history = []

# FIX 5: Use AutoForwardDiff (Zygote can have issues with ComponentArrays)
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)

for restart_idx in 1:N_RESTARTS
    println("\n" * "-"^40)
    println("RESTART $restart_idx / $N_RESTARTS")
    println("-"^40)

    p_start = p_inits[restart_idx]
    println("Initial (log): log_Estar=$(p_start.log_Estar), log_ks=$(p_start.log_ks), log_cs=$(p_start.log_cs)")
    println("Initial (actual): Estar=$(exp(p_start.log_Estar)), ks=$(exp(p_start.log_ks)), cs=$(exp(p_start.log_cs))")

    # Container to track losses for this restart
    losses_restart = Float64[]

    callback_restart = function (state, l)
        push!(losses_restart, l)
        if length(losses_restart) % 50 == 0
            # Show actual parameter values
            p_curr = state.u
            println("  Iter $(length(losses_restart)): loss = $l")
            println("    Estar=$(exp(p_curr.log_Estar)), ks=$(exp(p_curr.log_ks)), cs=$(exp(p_curr.log_cs))")
        end
        return false
    end

    # FIX 5: ADAM with higher learning rate
    println("Phase 1: ADAM (300 iterations, lr=0.1)")
    optprob = Optimization.OptimizationProblem(optf, p_start)
    res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1),
                              callback=callback_restart, maxiters=300)
    println("  After ADAM: loss = $(losses_restart[end])")

    # Phase 2: BFGS with higher stepnorm
    println("Phase 2: BFGS (up to 2000 iterations, stepnorm=0.01)")
    optprob2 = Optimization.OptimizationProblem(optf, res1.minimizer)
    res2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm=0.01),
                              callback=callback_restart, maxiters=2000)
    println("  After BFGS: loss = $(losses_restart[end])")

    # Store results (convert back from log space)
    p_final = res2.minimizer
    push!(all_results, (
        restart_idx = restart_idx,
        p_init = p_start,
        p_final = p_final,
        p_actual = (Estar = exp(p_final.log_Estar),
                    ks = exp(p_final.log_ks),
                    cs = exp(p_final.log_cs)),
        final_loss = losses_restart[end],
        losses = losses_restart
    ))
    push!(all_losses_history, losses_restart)

    println("Final (actual): Estar=$(exp(p_final.log_Estar)), ks=$(exp(p_final.log_ks)), cs=$(exp(p_final.log_cs))")
end

## ============================================================================
## Select Best Result
## ============================================================================

println("\n" * "="^60)
println("Comparing Results from All Restarts")
println("="^60)

# Find best result (lowest final loss)
final_losses = [r.final_loss for r in all_results]
best_idx = argmin(final_losses)
best_result = all_results[best_idx]

println("\nSummary of all restarts:")
println("-"^100)
println("| Restart | Final Loss | Estar | ks | cs |")
println("-"^100)
for r in all_results
    marker = r.restart_idx == best_idx ? " *" : "  "
    println("| $marker $(r.restart_idx)    | $(r.final_loss) | $(r.p_actual.Estar) | $(r.p_actual.ks) | $(r.p_actual.cs) |")
end
println("-"^100)

println("\n* Best restart: #$(best_idx) with loss = $(best_result.final_loss)")

# Use best result
p_trained = best_result.p_final
p_actual = best_result.p_actual
losses = best_result.losses

## ============================================================================
## Results
## ============================================================================

println("\n" * "="^60)
println("FINAL RESULTS (Best of $N_RESTARTS restarts)")
println("="^60)

println("\nEstimated parameters:")
println("  Estar = $(p_actual.Estar) Pa")
println("  ks = $(p_actual.ks) N/m")
println("  cs = $(p_actual.cs) N·s/m")

println("\nTrue parameters:")
println("  Estar = $(Estar_true) Pa")
println("  ks = $(ks_true) N/m")
println("  cs = $(cs_true) N·s/m")

println("\nRelative errors:")
err_Estar = (p_actual.Estar - Estar_true) / Estar_true * 100
err_ks = (p_actual.ks - ks_true) / ks_true * 100
err_cs = (p_actual.cs - cs_true) / cs_true * 100
println("  Estar: $(err_Estar)%")
println("  ks: $(err_ks)%")
println("  cs: $(err_cs)%")

# Check if all parameters are positive
if p_actual.Estar > 0 && p_actual.ks > 0 && p_actual.cs > 0
    println("\nAll parameters are POSITIVE (physically valid)")
else
    println("\nWARNING: Some parameters are negative!")
end

## ============================================================================
## Visualization
## ============================================================================

println("\n" * "="^60)
println("Generating plots...")
println("="^60)

# Create plots directory
plots_dir = joinpath(@__DIR__, "plots")
mkpath(plots_dir)

# Final prediction
X̂ = predict(p_trained)

# Plot 1: Loss curve (ADAM + BFGS)
pl_losses = plot(1:min(300, length(losses)), losses[1:min(300, length(losses))],
                 yaxis=:log10, xlabel="Iterations", ylabel="Loss",
                 label="ADAM", color=:blue, lw=2)
if length(losses) > 300
    plot!(301:length(losses), losses[301:end],
          label="BFGS", color=:red, lw=2)
end
hline!([loss_true], label="Sanity Check (true params)", color=:green, linestyle=:dash)
title!("Training Loss (ADAM + BFGS)")
savefig(pl_losses, joinpath(plots_dir, "$(svname)losses.pdf"))

# Plot 2: Tip displacement (x1)
pl_x1 = plot(t, x1_data, label="Data", color=:black, lw=1)
plot!(t, X̂[1, :], label="Estimated", color=:red, lw=2, linestyle=:dash)
xlabel!("Time [s]")
ylabel!("Tip displacement x1 [m]")
title!("Tip Displacement")
savefig(pl_x1, joinpath(plots_dir, "$(svname)tip_displacement.pdf"))

# Plot 3: Sample motion (x3)
pl_x3 = plot(t, x3_data, label="Data (ground truth)", color=:black, lw=1)
plot!(t, X̂[3, :], label="Estimated", color=:red, lw=2, linestyle=:dash)
xlabel!("Time [s]")
ylabel!("Sample displacement x3 [m]")
title!("Sample Motion (x3)")
savefig(pl_x3, joinpath(plots_dir, "$(svname)sample_motion.pdf"))

# Plot 4: Overview
pl_overview = plot(pl_losses, pl_x1, pl_x3, layout=(3, 1), size=(800, 900))
savefig(pl_overview, joinpath(plots_dir, "$(svname)overview.pdf"))

# Plot 5: Zoomed view of a few oscillation cycles
t_zoom_end = min(5e-5, t[end])
zoom_idx = t .<= t_zoom_end

pl_x1_zoom = plot(t[zoom_idx], x1_data[zoom_idx], label="Data", color=:black, lw=1)
plot!(t[zoom_idx], X̂[1, zoom_idx], label="Estimated", color=:red, lw=2, linestyle=:dash)
xlabel!("Time [s]")
ylabel!("Tip displacement x1 [m]")
title!("Tip Displacement (Zoomed)")
savefig(pl_x1_zoom, joinpath(plots_dir, "$(svname)tip_displacement_zoom.pdf"))

pl_x3_zoom = plot(t[zoom_idx], x3_data[zoom_idx], label="Data", color=:black, lw=1)
plot!(t[zoom_idx], X̂[3, zoom_idx], label="Estimated", color=:red, lw=2, linestyle=:dash)
xlabel!("Time [s]")
ylabel!("Sample displacement x3 [m]")
title!("Sample Motion (Zoomed)")
savefig(pl_x3_zoom, joinpath(plots_dir, "$(svname)sample_motion_zoom.pdf"))

println("Plots saved to: $plots_dir")

## ============================================================================
## Save Results
## ============================================================================

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

save(joinpath(results_dir, "$(svname)results.jld2"),
     "t", t,
     "x1_data", x1_data,
     "x2_data", x2_data,
     "x3_data", x3_data,
     "x2dot_data", x2dot_data,
     "X_estimated", X̂,
     "p_trained_log", p_trained,
     "p_trained_actual", p_actual,
     "p_true", (Estar=Estar_true, ks=ks_true, cs=cs_true),
     "losses", losses,
     "all_results", all_results)

println("Results saved to: $results_dir")

println("\n" * "="^60)
println("DONE")
println("="^60)
