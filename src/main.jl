
# solver = DESPOTSolver(
#                 bounds=IndependentBounds(lower, upper),
#                 tree_in_info=false)

using POMDPGifs # to make gifs
using Cairo # for making/saving the gif
using RockSample
using POMDPs
using POMDPTools
using BasicPOMCP
using ParticleFilters
using Random
using FileIO
using Statistics
using ProgressMeter
include("utils.jl")
include("main_pomcp.jl")

const n_experiments = 2
pomcp_data = Array{Tuple,1}(undef, n_experiments)
despot_data = Array{Tuple,1}(undef, n_experiments)

output_filename = "results/results.out"
close(open(output_filename, "w"))
delete_old_files()

for i in 1:n_experiments
    env = get_enviroment()
    open(output_filename, "a") do io
        write_and_print(io, "Running Simulation #$i")
        write_and_print(io, "Rocks Position -> " * string(env.rocks_positions))
    end
    pomcp_data[i] = run_one_experiment_pomcp(deepcopy(env), i)
end
compute_final_results(pomcp_data)