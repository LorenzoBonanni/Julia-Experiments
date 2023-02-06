function plot_return(ret:: Vector, algo::String)
    histogram(ret)
    StatsPlots.savefig("results/return/hist_$algo.png")
end

function plot_action_dist(actions:: Vector{String}, i::Int, algo::String)
    c = countmap(actions)

    Plots.bar(string.(collect(keys(c))), collect(values(c)))
    Plots.savefig("results/actions/actDist_$(algo)_$(i).png")
end