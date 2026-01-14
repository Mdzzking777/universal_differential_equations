# 查看.jld2文件内容的工具
# 使用方法: julia --project=. inspect_results.jl

# 自动切换到脚本所在目录
cd(@__DIR__)

using JLD2, FileIO, ComponentArrays, Printf, Statistics, Dates

println("="^70)
println("JLD2 RESULTS FILE INSPECTOR")
println("="^70)
println()

# 加载文件
filepath = "results/AFM_Scenario_results.jld2"
println("Loading: $filepath")
results = load(filepath)

# 显示所有键
println("\n" * "="^70)
println("FILE CONTENTS (Keys)")
println("="^70)
all_keys = keys(results)
for (i, key) in enumerate(sort(collect(all_keys)))
    println("  [$i] $key")
end

# 显示每个变量的详细信息
println("\n" * "="^70)
println("VARIABLE DETAILS")
println("="^70)

for key in sort(collect(all_keys))
    value = results[key]
    println("\n[$key]")
    println("  Type: $(typeof(value))")

    if value isa Number
        println("  Value: $value")
    elseif value isa AbstractArray
        println("  Size: $(size(value))")
        println("  Length: $(length(value))")
        if length(value) > 0
            if eltype(value) <: Number
                println("  Range: $(minimum(value)) to $(maximum(value))")
                println("  First 3 elements: $(value[1:min(3, length(value))])")
            else
                println("  First element: $(value[1])")
            end
        end
    elseif value isa Vector
        println("  Length: $(length(value))")
        if length(value) <= 5
            println("  All elements: $value")
        else
            println("  First 3: $(value[1:3])")
            println("  Last 3: $(value[end-2:end])")
        end
    else
        println("  Value: $value")
    end
end

# 特别显示优化参数
println("\n" * "="^70)
println("OPTIMIZED PARAMETERS (theta_optimal)")
println("="^70)

θ = results["theta_optimal"]
println("\nGlobal Parameters:")
println("  θ[1] = ks    = $(θ[1]) N/m")
println("  θ[2] = cs    = $(θ[2]) kg/s")
println("  θ[3] = Estar = $(θ[3]) Pa")

println("\nLocal Variables (y0 for each segment):")
println("  Number of segments: $(length(θ) - 3)")
println("  θ[4:6]   (first 3)  = $(θ[4:6])")
println("  θ[end-2:end] (last 3) = $(θ[end-2:end])")

# 显示Ground Truth对比
println("\n" * "="^70)
println("GROUND TRUTH vs RECOVERED")
println("="^70)

ks_true = results["ks_true"]
cs_true = results["cs_true"]
Estar_true = results["Estar_true"]

println("\nParameter    | True Value      | Recovered       | Error")
println("-"^70)
println(@sprintf("ks (N/m)     | %-15.6e | %-15.6e | %.2f%%",
        ks_true, θ[1], abs(θ[1]-ks_true)/ks_true*100))
println(@sprintf("cs (kg/s)    | %-15.6e | %-15.6e | %.2f%%",
        cs_true, θ[2], abs(θ[2]-cs_true)/cs_true*100))
println(@sprintf("Estar (Pa)   | %-15.6e | %-15.6e | %.2f%%",
        Estar_true, θ[3], abs(θ[3]-Estar_true)/Estar_true*100))

# 显示数据统计
println("\n" * "="^70)
println("DATA STATISTICS")
println("="^70)

t_data = results["t_data"]
x_data = results["x_data"]
y_data = results["y_data"]
x_recon = results["x_reconstructed"]
y_recon = results["y_reconstructed"]

println("\nTime series:")
println("  Duration: $(t_data[1]) to $(t_data[end]) s")
println("  Points: $(length(t_data))")
println("  Sampling rate: $(@sprintf("%.2f kHz", 1/mean(diff(t_data))/1000))")

println("\nTip displacement (x):")
println("  Range: $(minimum(x_data)*1e9) to $(maximum(x_data)*1e9) nm")
println("  Mean: $(mean(x_data)*1e9) nm")
println("  Reconstruction RMSE: $(sqrt(mean((x_data .- x_recon).^2))*1e9) nm")

println("\nSample motion (y):")
println("  Range: $(minimum(y_data)*1e9) to $(maximum(y_data)*1e9) nm")
println("  Mean: $(mean(y_data)*1e9) nm")

# 计算contact region误差
contact_segs = results["contact_segments"]
contact_points = sum(seg.stop - seg.start + 1 for seg in contact_segs)

println("\nContact segments:")
println("  Number of segments: $(length(contact_segs))")
println("  Total contact points: $contact_points ($(@sprintf("%.2f%%", contact_points/length(t_data)*100)))")

# 优化历史
losses = results["losses"]
println("\nOptimization:")
println("  Total iterations: $(length(losses))")
println("  Initial loss: $(@sprintf("%.6e", losses[1]))")
println("  Final loss: $(@sprintf("%.6e", losses[end]))")
println("  Loss reduction: $(@sprintf("%.2f%%", (losses[1]-losses[end])/losses[1]*100))")

println("\n" * "="^70)
println("END OF INSPECTION")
println("="^70)

# 保存摘要到文本文件
using Statistics

summary_file = "results/results_summary.txt"
open(summary_file, "w") do io
    println(io, "AFM DMT-KV PARAMETER RECOVERY RESULTS")
    println(io, "="^70)
    println(io, "Generated: $(now())")
    println(io, "\nRECOVERED PARAMETERS:")
    println(io, "  ks    = $(θ[1]) N/m")
    println(io, "  cs    = $(θ[2]) kg/s")
    println(io, "  Estar = $(θ[3]) Pa")
    println(io, "\nGROUND TRUTH:")
    println(io, "  ks    = $(ks_true) N/m")
    println(io, "  cs    = $(cs_true) kg/s")
    println(io, "  Estar = $(Estar_true) Pa")
    println(io, "\nERRORS:")
    println(io, "  ks    = $(@sprintf("%.2f%%", abs(θ[1]-ks_true)/ks_true*100))")
    println(io, "  cs    = $(@sprintf("%.2f%%", abs(θ[2]-cs_true)/cs_true*100))")
    println(io, "  Estar = $(@sprintf("%.2f%%", abs(θ[3]-Estar_true)/Estar_true*100))")
end

println("\n✓ Summary saved to: $summary_file")
println("  (You can open this .txt file in VSCode)")
