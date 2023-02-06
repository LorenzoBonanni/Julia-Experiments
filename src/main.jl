using POMDPGifs # to make gifs
using Cairo # for making/saving the gif
using RockSample, POMDPs, POMDPTools, ParticleFilters, Random, FileIO, Statistics, 
     ProgressMeter, CSV, DataFrames, StatsPlots, DataStructures, Dates, StatsBase
include("utils.jl")
include("main_pomcp.jl")
include("main_despot.jl")
include("heuristic.jl")
include("plots.jl")

function get_enviroment()::RockSamplePOMDP
    # rocks = [Tuple{Int,Int}(rand(rand_noise_generator_for_env, 1:map_size, 2)) for _ in 1:num_rock]
    rocks = [(1,9), (6,3), (2,5), (6,6)]
    pomdp = RockSamplePOMDP{num_rock}(
        map_size=(map_size, map_size),
        rocks_positions=rocks,
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=10.0,
        bad_rock_penalty=-10.0,
        sensor_use_penalty=0.0,
        step_penalty=0.0,
    )
    return pomdp
end

const n_particle = 32768 # 2^15
const n_sim = n_particle
const max_steps = 100
const save_steps = true
const save_gif = true
const n_experiments = 50
const num_rock = 4
const map_size = 12
rand_noise_generator_seed_for_env = rand(UInt32)
rand_noise_generator_seed_for_sim = rand(UInt32)
rand_noise_generator_seed_for_planner = rand(UInt32)
rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)
rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)
rand_noise_generator_for_planner = MersenneTwister(rand_noise_generator_seed_for_planner)

pomcp_data = Array{Tuple,1}(undef, n_experiments)
despot_data = Array{Tuple,1}(undef, n_experiments)
despot_data_informed = Array{Tuple,1}(undef, n_experiments)

output_filename_pomcp = "results/results_POMCP.out"
output_filename_despot = "results/results_DESPOT.out"
output_filename_despot_informed = "results/results_DESPOT_INFORMED.out"
delete_old_files()

start = Dates.format(now(), "HH:MM")

for i in 1:n_experiments
    env = get_enviroment()

    println("Computing UpperBound...")
    compute_upperbound(deepcopy(env))
    open(output_filename_pomcp, "a") do io
        write_and_print(io, "Running Simulation #$i")
        write_and_print(io, "Rocks Position -> " * string(env.rocks_positions))
    end
    open(output_filename_despot, "a") do io
        write(io, "Running Simulation #$i" * "\n")
        write(io, "Rocks Position -> " * string(env.rocks_positions) * "\n")
    end
    open(output_filename_despot_informed, "a") do io
        write(io, "Running Simulation #$i" * "\n")
        write(io, "Rocks Position -> " * string(env.rocks_positions) * "\n")
    end
    pomcp_data[i] = run_one_experiment_pomcp(deepcopy(env), i)
    despot_data[i] = run_one_experiment_despot(deepcopy(env), i)
    despot_data_informed[i] = run_one_experiment_despot_informed(deepcopy(env), i)
end
compute_final_results(pomcp_data, "POMCP")
compute_final_results(despot_data, "DESPOT")
compute_final_results(despot_data_informed, "DESPOT_INFORMED")
println("\nInitial Time: $(start)\n Final Time: $(Dates.format(now(), "HH:MM"))")