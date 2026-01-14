# Quick script to read recovered parameters from results file

# 自动切换到脚本所在目录（确保能找到results文件夹）
cd(@__DIR__)

# Activate project environment
using Pkg; Pkg.activate(".");

using JLD2, FileIO, ComponentArrays, Printf

# Load results
results = load("results/AFM_Scenario_results.jld2")

# Extract optimal parameters
θ_opt = results["theta_optimal"]

# Access as ComponentArray
ks_opt = θ_opt[1]
cs_opt = θ_opt[2]
Estar_opt = θ_opt[3]
y0_segments = θ_opt[4:end]

println("="^60)
println("RECOVERED PARAMETERS")
println("="^60)
println()
println("Global parameters (unknowns we wanted to recover):")
println("  ks     = $(ks_opt) N/m          (surface stiffness)")
println("  cs     = $(cs_opt) kg/s         (surface damping)")
println("  Estar  = $(Estar_opt) Pa           (elastic modulus)")
println()
println("Ground truth (for comparison):")
println("  ks     = $(results["ks_true"]) N/m")
println("  cs     = $(results["cs_true"]) kg/s")
println("  Estar  = $(results["Estar_true"]) Pa")
println()
println("Errors:")
println("  ks     = $(@sprintf("%.2f%%", abs(ks_opt - results["ks_true"])/results["ks_true"]*100))")
println("  cs     = $(@sprintf("%.2f%%", abs(cs_opt - results["cs_true"])/results["cs_true"]*100))")
println("  Estar  = $(@sprintf("%.2f%%", abs(Estar_opt - results["Estar_true"])/results["Estar_true"]*100))")
println()
println("Local variables (y0 for each segment):")
println("  Number of segments: $(length(y0_segments))")
println("  y0_segments[1]   = $(y0_segments[1]) m")
println("  y0_segments[144] = $(y0_segments[144]) m")
println("  y0_segments[288] = $(y0_segments[288]) m")
println()
println("Total optimized variables: $(length(θ_opt))")
println("="^60)