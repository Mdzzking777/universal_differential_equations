"""
Quick runner script for AFM parameter recovery
Checks dependencies and runs the main scenario
"""

println("="^60)
println("AFM DMT-KV Parameter Recovery - Quick Runner")
println("="^60)

## Step 1: Check if trajectory.csv exists
data_path = joinpath(dirname(@__DIR__), "trajectory.csv")

if !isfile(data_path)
    println("\n❌ Error: trajectory.csv not found!")
    println("\nPlease run the Python script first:")
    println("  cd ..")
    println("  python generate_trajectory_DMT_KV.py")
    println("\nThis will generate the required trajectory.csv file.")
    exit(1)
else
    println("✓ Found trajectory.csv")
end

## Step 2: Check if Julia environment is set up
println("\n[1/3] Checking Julia environment...")
using Pkg

cd(@__DIR__)
if !isfile("Project.toml")
    println("❌ Project.toml not found!")
    exit(1)
end

println("  Activating environment...")
Pkg.activate(".")

println("  Instantiating packages (this may take a while on first run)...")
Pkg.instantiate()

println("✓ Environment ready")

## Step 3: Run the main scenario
println("\n[2/3] Running scenario_afm.jl...")
println("-"^60)

include("scenario_afm.jl")

println("\n[3/3] Done!")
println("\n" * "="^60)
println("SUCCESS! Check the following:")
println("  - plots/ folder for visualizations")
println("  - results/ folder for saved data")
println("="^60)
