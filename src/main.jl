using POMDPs, POMDPModels,POMDPModelTools, POMDPSimulators, 
ARDESPOT, RockSample, Random, D3Trees, ParticleFilters
using POMDPGifs # to make gifs
using Cairo # for making/saving the gif
include("utils.jl")

const num_rock = 11
const map_size = 11 
rng = MersenneTwister(1);
rocks = [Tuple{Int,Int}(rand(rng,1:map_size, 2)) for _ in 1:num_rock]
pomdp = RockSamplePOMDP{num_rock}(
                        map_size=(map_size, map_size),
                        rocks_positions = rocks,
                        sensor_efficiency=20.0,
                        discount_factor=0.95, 
                        good_rock_reward=10.0,
                        bad_rock_penalty=-10.0,
                        sensor_use_penalty=0.0,
                        step_penalty=0.0)

solver = DESPOTSolver(
                bounds=IndependentBounds(lower, upper),
                tree_in_info=false)
planner = solve(solver, pomdp)

filter = POMDPs.updater(planner)

rewards = []
for (b,a,o,r) in stepthrough(pomdp, planner, filter, "b,a,o,r", max_steps=50)
    println("action $a was taken,")
    println("reward $r was obtained")
    println("and observation $o was received.\n")
    push!(rewards, r)
end

println("Cumulative Sum $(sum(rewards))")

# sim = GifSimulator(filename="test.gif", max_steps=30)
# simulate(sim, pomdp, planner)