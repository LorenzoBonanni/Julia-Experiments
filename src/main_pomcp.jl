const num_rock = 4
const map_size = 12
const n_particle = 32768 # 2^15
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