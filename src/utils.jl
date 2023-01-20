# values = load("upper_bound.jld2")["data"]


# function upper(pomdp::POMDP, b::ScenarioBelief)
#     scenarios = last.(b.scenarios)
#     scenarios_idx = stateindex.([pomdp], scenarios)
#     len = first(last(b.scenarios))
#     mdp_values = getindex(values, scenarios_idx)
#     return sum(mdp_values)/len
# end

# # 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT, ..., K sample kth rock
# always_left = FunctionPolicy(b->return RockSample.BASIC_ACTIONS_DICT[:east])
# lower = DefaultPolicyLB(always_left)

function write_and_print(io::IOStream, string_to_be_written_and_printed::String)
    write(io, string_to_be_written_and_printed * "\n")
    println(string_to_be_written_and_printed)
end

function save_experiment_data(hist::HistoryIterator, env::RockSamplePOMDP)
    makegif(env,
        hist,
        filename="results/gif/pomcp_$i.gif",
        show_progress=true,
    )
end