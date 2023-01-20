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

const num_rock = 4
const map_size = 12
const n_particle = 32768 # 2^15
const n_experiments = 2
const max_steps = 1000
const save_steps = true
const save_gif = true

rand_noise_generator_seed_for_env = rand(UInt32)
rand_noise_generator_seed_for_sim = rand(UInt32)
rand_noise_generator_seed_for_planner = rand(UInt32)
rand_noise_generator_for_sim = MersenneTwister(rand_noise_generator_seed_for_sim)
rand_noise_generator_for_env = MersenneTwister(rand_noise_generator_seed_for_env)

function get_enviroment()::RockSamplePOMDP
    rocks = [Tuple{Int,Int}(rand(rand_noise_generator_for_env, 1:map_size, 2)) for _ in 1:num_rock]
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

function save_history(hist::SimHistory, i::Int)
    output_filename = "results/history/hist_$i.out"
    ACTIONS = Dict(value => string(key) for (key, value) in RockSample.BASIC_ACTIONS_DICT)
    open(output_filename, "w") do io
        write(io, "S, A, R, SP, O" * "\n")
        @showprogress 1 "Saving History..." for (s, a, r, sp, o, ai, ui) in eachstep(hist, "(s, a, r, sp, o, action_info, update_info)")
            action = get(ACTIONS, a, a)
            write(io, "$s, $action, $r, $sp, $o" * "\n")
            write(io, "Action Info: $ai" * "\n")
            write(io, "Update Info: $ui" * "\n")
        end
    end
end

function save_experiment_data(hist::SimHistory, env::RockSamplePOMDP, i::Int)
    output_filename = "results/results.out"
    if save_gif
        makegif(
            env,
            hist,
            filename="results/gif/pomcp_$i.gif",
            show_progress=true,
        )
    end
    if save_steps
        save_history(hist, i)
    end

    open(output_filename, "a") do io
        write_and_print(io, "#Steps POMCP -> " * string(n_steps(hist)))
        write_and_print(io, "Cumulative Discounted Return POMCP -> " * string(discounted_reward(hist)) * "\n")
    end
end

function run_one_experiment_pomcp(env::RockSamplePOMDP, i::Int)
    pf = UnweightedParticleFilter(env, n_particle, rand_noise_generator_for_sim)

    solver = POMCPSolver(
        estimate_value=RolloutEstimator(RandomPolicy(env, rng=rand_noise_generator_for_sim)),
        max_depth=100,
        c=1.0,
        tree_queries=10000,
        rng=MersenneTwister(rand_noise_generator_seed_for_planner)
    )

    policy = solve(solver, env)
    sim = HistoryRecorder(
        rng=rand_noise_generator_for_sim,
        max_steps=max_steps,
        show_progress=true
    )

    t0 = time()
    hist = simulate(sim, env, policy, pf)
    elapsed = time() - t0
    open(output_filename, "a") do io
        write(io, "Planner Seed -> $(solver.rng.seed[1])" * "\n")
        write(io, "Filter Seed -> $(pf.rng.seed[1])" * "\n")
        write(io, "Rollout Seed -> $(solver.estimate_value.solver.rng.seed[1])" * "\n")
        write(io, "Simulator Seed -> $(sim.rng.seed[1])" * "\n")
        write_and_print(io, "Elapsed Time POMCP -> " * string(elapsed))
    end

    save_experiment_data(hist, env, i)
    return discounted_reward(hist), convert(Float64, n_steps(hist))
end

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