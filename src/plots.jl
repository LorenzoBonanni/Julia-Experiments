function plot_return(ret:: Vector, algo::String)
    histogram(ret)
    StatsPlots.savefig("results/hist_$algo.png")
end

function plot_action_dist(actions:: Vector{String}, i::Int)
    c = DataStructures.counter(actions)
    Plots.bar(string.(collect(keys(c))), collect(values(c)))
    Plots.savefig("results/actDist_$algo_$i.png")
end