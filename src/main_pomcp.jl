using BasicPOMCP

function run_one_experiment_pomcp(env::RockSamplePOMDP, i::Int)
    solver = POMCPSolver(
        estimate_value=RolloutEstimator(RandomPolicy(env, rng=rand_noise_generator_for_sim)),
        max_depth=100,
        c=1.0,
        tree_queries=n_sim,
        rng=rand_noise_generator_for_planner
    )

    policy = solve(solver, env)
    pf = updater(policy)
    sim = HistoryRecorder(
        rng=rand_noise_generator_for_sim,
        max_steps=max_steps,
        show_progress=true
    )

    t0 = time()
    hist = simulate(sim, env, policy, pf)
    elapsed = time() - t0
    open(output_filename_pomcp, "a") do io
        write(io, "Planner Seed -> $(solver.rng.seed[1])" * "\n")
        write(io, "Filter Seed -> $(pf.rng.seed[1])" * "\n")
        write(io, "Rollout Seed -> $(solver.estimate_value.solver.rng.seed[1])" * "\n")
        write(io, "Simulator Seed -> $(sim.rng.seed[1])" * "\n")
        write_and_print(io, "Elapsed Time POMCP -> " * string(elapsed))
    end

    save_experiment_data(hist, env, i, "POMCP")
    return discounted_reward(hist), undiscounted_reward(hist), convert(Float64, n_steps(hist)), elapsed
end