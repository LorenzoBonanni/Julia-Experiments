using POMDPModels, RockSample, Random, DiscreteValueIteration, POMDPModelTools, POMDPs, FileIO

const num_rock = 11
const map_size = 11 
rocks = [Tuple{Int,Int}(rand(MersenneTwister(0),1:map_size, 2)) for _ in 1:num_rock]
pomdp = RockSamplePOMDP{num_rock}(
                        map_size=(map_size, map_size),
                        rocks_positions = rocks,
                        sensor_efficiency=20.0,
                        discount_factor=0.95, 
                        good_rock_reward=10.0,
                        bad_rock_penalty=-10.0,
                        sensor_use_penalty=0.0,
                        step_penalty=0.0)

vals = RockSample.rs_mdp_utility(pomdp)
save("upper_bound.jld2", "data", vals)
