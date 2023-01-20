values = load("upper_bound.jld2")["data"]


function upper(pomdp::POMDP, b::ScenarioBelief)
    scenarios = last.(b.scenarios)
    scenarios_idx = stateindex.([pomdp], scenarios)
    len = first(last(b.scenarios))
    mdp_values = getindex(values, scenarios_idx)
    return sum(mdp_values)/len
end

# 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT, ..., K sample kth rock
always_left = FunctionPolicy(b->return RockSample.BASIC_ACTIONS_DICT[:east])
lower = DefaultPolicyLB(always_left)