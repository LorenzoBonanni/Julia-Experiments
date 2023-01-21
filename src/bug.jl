using RockSample
using POMDPs
using POMDPTools
using BasicPOMCP
using Random
using ParticleFilters
using LinearAlgebra
using StaticArrays

const n_particle = 32768 # 2^15
rand_noise_generator_for_sim = MersenneTwister(2980164632)
rand_noise_generator_seed_for_planner = MersenneTwister(941564507)
ACTIONS = Dict(value => string(key) for (key, value) in RockSample.BASIC_ACTIONS_DICT)

rocks = [[10, 1], [11, 6], [9, 6], [2, 5]]
env = RockSamplePOMDP{4}(
        map_size=(12, 12),
        rocks_positions=rocks,
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=10.0,
        bad_rock_penalty=-10.0,
        sensor_use_penalty=0.0,
        step_penalty=0.0,
        init_pos=[7, 3]
    )

pf = UnweightedParticleFilter(env, n_particle, rand_noise_generator_for_sim)

solver = POMCPSolver(
    estimate_value=RolloutEstimator(RandomPolicy(env, rng=rand_noise_generator_for_sim)),
    max_depth=100,
    c=1.0,
    tree_queries=10000,
    rng=rand_noise_generator_seed_for_planner
)
policy = solve(solver, env)

function POMDPs.initialstate(pomdp::RockSamplePOMDP{K}) where K
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for i in 1:2^K
        states[i] = RSState{4}([7, 3], Bool[0, 0, 0, 0])
    end
    return SparseCat(states, probs)
end

for (s, a, o, r) in stepthrough(env, policy, "s,a,o,r", max_steps=1)
    println("in state $s" * "\n")
    println("took action $(ACTIONS[a])" * "\n")
    println("received observation $o and reward $r")
end
