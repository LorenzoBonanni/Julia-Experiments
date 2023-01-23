
function run_one_experiment_despot(env::RockSamplePOMDP, i::Int)
    pf = BasicParticleFilter(env, ImportanceResampler(n_particle), n_particle, rand_noise_generator_for_sim)

    solver = DESPOTSolver(
        bounds=IndependentBounds(-20, 0, check_terminal=true),
        lambda=0.0,
        D=50,
        xi=0.95,
        max_trials=10000,
        K=500
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
    open(output_filename_despot, "a") do io
        write(io, "Planner Seed -> $(solver.rng.seed[1])" * "\n")
        write(io, "Filter Seed -> $(pf.rng.seed[1])" * "\n")
        write(io, "Simulator Seed -> $(sim.rng.seed[1])" * "\n")
        write_and_print(io, "Elapsed Time DESPOT -> " * string(elapsed))
    end

    save_experiment_data(hist, env, i, "DESPOT")
    return discounted_reward(hist), convert(Float64, n_steps(hist)), elapsed
end

function run_one_experiment_despot_informed(env::RockSamplePOMDP, i::Int)
    pf = BasicParticleFilter(env, ImportanceResampler(n_particle), n_particle, rand_noise_generator_for_sim)

    solver = DESPOTSolver(
        bounds=IndependentBounds(lower, upper),
        lambda=0.0,
        D=50,
        xi=0.95,
        max_trials=10000,
        K=500
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
    open(output_filename_despot, "a") do io
        write(io, "Planner Seed -> $(solver.rng.seed[1])" * "\n")
        write(io, "Filter Seed -> $(pf.rng.seed[1])" * "\n")
        write(io, "Simulator Seed -> $(sim.rng.seed[1])" * "\n")
        write_and_print(io, "Elapsed Time DESPOT_INFORMED -> " * string(elapsed))
    end

    save_experiment_data(hist, env, i, "DESPOT_INFORMED")
    return discounted_reward(hist), convert(Float64, n_steps(hist)), elapsed
end