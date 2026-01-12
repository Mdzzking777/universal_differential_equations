"""
Analysis and Visualization of AFM Parameter Recovery Results
Load saved results and create detailed comparison plots
"""

using JLD2, FileIO
using Plots
using Statistics
using Printf
gr()

cd(@__DIR__)

println("="^60)
println("AFM Results Analysis")
println("="^60)

## Load results
results_file = joinpath(pwd(), "results", "AFM_Scenario_results.jld2")

if !isfile(results_file)
    error("Results file not found! Run scenario_afm.jl first.")
end

println("\nLoading results from $(results_file)...")
data = load(results_file)

t_data = data["t_data"]
x_data = data["x_data"]
y_data = data["y_data"]
x_reconstructed = data["x_reconstructed"]
y_reconstructed = data["y_reconstructed"]
contact_segments = data["contact_segments"]
θ_optimal = data["theta_optimal"]
losses = data["losses"]
ks_true = data["ks_true"]
cs_true = data["cs_true"]
Estar_true = data["Estar_true"]

println("  Loaded $(length(t_data)) time points")
println("  $(length(contact_segments)) contact segments")

## Extract contact regions
contact_mask = falses(length(t_data))
for seg in contact_segments
    contact_mask[seg.start:seg.stop] .= true
end

t_contact = t_data[contact_mask]
x_contact = x_data[contact_mask]
y_contact = y_data[contact_mask]
x_recon_contact = x_reconstructed[contact_mask]
y_recon_contact = y_reconstructed[contact_mask]

## Statistical Analysis
println("\n" * "="^60)
println("STATISTICAL ANALYSIS")
println("="^60)

# Parameter errors
ks_error = abs(θ_optimal.ks - ks_true) / ks_true * 100
cs_error = abs(θ_optimal.cs - cs_true) / cs_true * 100
Estar_error = abs(θ_optimal.Estar - Estar_true) / Estar_true * 100

println("\nParameter Recovery:")
println("  ks:    true=$(ks_true), recovered=$(@sprintf("%.6f", θ_optimal.ks)), error=$(@sprintf("%.2f", ks_error))%")
println("  cs:    true=$(@sprintf("%.3e", cs_true)), recovered=$(@sprintf("%.3e", θ_optimal.cs)), error=$(@sprintf("%.2f", cs_error))%")
println("  Estar: true=$(@sprintf("%.3e", Estar_true)), recovered=$(@sprintf("%.3e", θ_optimal.Estar)), error=$(@sprintf("%.2f", Estar_error))%")

# Reconstruction errors
x_mse = mean((x_contact .- x_recon_contact).^2)
y_mse = mean((y_contact .- y_recon_contact).^2)
x_rmse = sqrt(x_mse)
y_rmse = sqrt(y_mse)
x_nrmse = x_rmse / (maximum(x_contact) - minimum(x_contact))
y_nrmse = y_rmse / (maximum(y_contact) - minimum(y_contact))

println("\nReconstruction Errors (Contact Region):")
println("  x (observable):")
println("    RMSE:  $(@sprintf("%.3e", x_rmse)) m")
println("    NRMSE: $(@sprintf("%.2f", x_nrmse*100))%")
println("  y (hidden):")
println("    RMSE:  $(@sprintf("%.3e", y_rmse)) m")
println("    NRMSE: $(@sprintf("%.2f", y_nrmse*100))%")

# Correlation
x_corr = cor(x_contact, x_recon_contact)
y_corr = cor(y_contact, y_recon_contact)

println("\nCorrelation Coefficients:")
println("  x: $(@sprintf("%.6f", x_corr))")
println("  y: $(@sprintf("%.6f", y_corr))")

## Detailed Plots

println("\n" * "="^60)
println("GENERATING DETAILED PLOTS")
println("="^60)

# Plot 1: Residual Analysis
p1 = plot(title="Residual Analysis", layout=(2,1), size=(800, 600))

# x residuals
x_residuals = x_contact .- x_recon_contact
plot!(p1, t_contact .* 1e3, x_residuals .* 1e9,
     subplot=1, label="x residuals",
     xlabel="Time (ms)", ylabel="Residual (nm)",
     color=:blue, alpha=0.6)
hline!(p1, [0], subplot=1, color=:black, linestyle=:dash, label=nothing)

# y residuals
y_residuals = y_contact .- y_recon_contact
plot!(p1, t_contact .* 1e3, y_residuals .* 1e9,
     subplot=2, label="y residuals",
     xlabel="Time (ms)", ylabel="Residual (nm)",
     color=:red, alpha=0.6)
hline!(p1, [0], subplot=2, color=:black, linestyle=:dash, label=nothing)

savefig(p1, joinpath(pwd(), "plots", "AFM_Scenario_residuals.pdf"))
println("  ✓ Saved residuals plot")

# Plot 2: Scatter plot (predicted vs true)
p2 = plot(title="Predicted vs True", layout=(1,2), size=(1000, 400))

# x scatter
scatter!(p2, x_contact .* 1e9, x_recon_contact .* 1e9,
        subplot=1, label="x data", alpha=0.3, markersize=2,
        xlabel="True x (nm)", ylabel="Predicted x (nm)")
plot!(p2, [minimum(x_contact), maximum(x_contact)] .* 1e9,
     [minimum(x_contact), maximum(x_contact)] .* 1e9,
     subplot=1, color=:red, linestyle=:dash, label="Perfect fit", lw=2)
annotate!(p2, [(minimum(x_contact)*1e9, maximum(x_contact)*1e9,
          text("R² = $(@sprintf("%.4f", x_corr^2))", 10, :left))], subplot=1)

# y scatter
scatter!(p2, y_contact .* 1e9, y_recon_contact .* 1e9,
        subplot=2, label="y data", alpha=0.3, markersize=2, color=:red,
        xlabel="True y (nm)", ylabel="Predicted y (nm)")
plot!(p2, [minimum(y_contact), maximum(y_contact)] .* 1e9,
     [minimum(y_contact), maximum(y_contact)] .* 1e9,
     subplot=2, color=:blue, linestyle=:dash, label="Perfect fit", lw=2)
annotate!(p2, [(minimum(y_contact)*1e9, maximum(y_contact)*1e9,
          text("R² = $(@sprintf("%.4f", y_corr^2))", 10, :left))], subplot=2)

savefig(p2, joinpath(pwd(), "plots", "AFM_Scenario_scatter.pdf"))
println("  ✓ Saved scatter plot")

# Plot 3: Segment-by-segment comparison
n_segs = min(4, length(contact_segments))  # Show first 4 segments
p3 = plot(layout=(n_segs, 2), size=(1000, 300*n_segs),
         title=["Segment $i - x" for i in 1:n_segs, j in 1:2])

for (i, seg) in enumerate(contact_segments[1:n_segs])
    seg_idx = seg.start:seg.stop
    t_seg = t_data[seg_idx]

    # x comparison
    plot!(p3, t_seg .* 1e3, x_data[seg_idx] .* 1e9,
         subplot=2*(i-1)+1, label="True", color=:black, lw=2)
    plot!(p3, t_seg .* 1e3, x_reconstructed[seg_idx] .* 1e9,
         subplot=2*(i-1)+1, label="Pred", color=:red, linestyle=:dash, lw=2)
    plot!(p3, xlabel="Time (ms)", ylabel="x (nm)", subplot=2*(i-1)+1)

    # y comparison
    plot!(p3, t_seg .* 1e3, y_data[seg_idx] .* 1e9,
         subplot=2*i, label="True", color=:black, lw=2)
    plot!(p3, t_seg .* 1e3, y_reconstructed[seg_idx] .* 1e9,
         subplot=2*i, label="Pred", color=:blue, linestyle=:dash, lw=2)
    plot!(p3, xlabel="Time (ms)", ylabel="y (nm)", subplot=2*i)
end

savefig(p3, joinpath(pwd(), "plots", "AFM_Scenario_segments.pdf"))
println("  ✓ Saved segment comparison plot")

# Plot 4: Parameter comparison bar chart
p4 = plot(layout=(1,3), size=(1200, 400))

# ks
bar!(p4, ["True", "Recovered"], [ks_true, θ_optimal.ks],
    subplot=1, title="Surface Stiffness (ks)",
    ylabel="N/m", legend=false, color=[:blue, :red])

# cs
bar!(p4, ["True", "Recovered"], [cs_true, θ_optimal.cs] .* 1e6,
    subplot=2, title="Surface Damping (cs)",
    ylabel="µN·s/m", legend=false, color=[:blue, :red])

# Estar
bar!(p4, ["True", "Recovered"], [Estar_true, θ_optimal.Estar] ./ 1e6,
    subplot=3, title="Elastic Modulus (E*)",
    ylabel="MPa", legend=false, color=[:blue, :red])

savefig(p4, joinpath(pwd(), "plots", "AFM_Scenario_parameters.pdf"))
println("  ✓ Saved parameter comparison plot")

## Summary Report
println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("\nOptimization:")
println("  - Total iterations: $(length(losses))")
println("  - Final loss: $(@sprintf("%.6e", losses[end]))")
println("  - Loss reduction: $(@sprintf("%.2f", (losses[1] - losses[end])/losses[1] * 100))%")

println("\nParameter Recovery Quality:")
if maximum([ks_error, cs_error, Estar_error]) < 5.0
    println("  ✓ EXCELLENT (<5% error)")
elseif maximum([ks_error, cs_error, Estar_error]) < 10.0
    println("  ✓ GOOD (<10% error)")
elseif maximum([ks_error, cs_error, Estar_error]) < 20.0
    println("  ⚠ ACCEPTABLE (<20% error)")
else
    println("  ✗ POOR (>20% error)")
end

println("\nState Reconstruction Quality:")
if y_nrmse < 0.05
    println("  ✓ EXCELLENT (<5% NRMSE)")
elseif y_nrmse < 0.10
    println("  ✓ GOOD (<10% NRMSE)")
elseif y_nrmse < 0.20
    println("  ⚠ ACCEPTABLE (<20% NRMSE)")
else
    println("  ✗ POOR (>20% NRMSE)")
end

println("\n" * "="^60)
println("Analysis complete!")
println("  - Generated 4 additional plots in ./plots/")
println("="^60)
