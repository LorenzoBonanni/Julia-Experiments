

function delete_old_files()
    foreach(rm, filter(endswith(".out"), readdir("results/", join=true)))
    foreach(rm, readdir("results/history", join=true))
    foreach(rm, readdir("results/gif", join=true))
end

function write_and_print(io::IOStream, string_to_be_written_and_printed::String)
    write(io, string_to_be_written_and_printed * "\n")
    println(string_to_be_written_and_printed)
end

function save_history(hist::SimHistory, i::Int, algo::String)
    output_filename = "results/history/hist_$(algo)_$i.out"
    ACTIONS = Dict(value => string(key) for (key, value) in RockSample.BASIC_ACTIONS_DICT)
    open(output_filename, "w") do io
        write(io, "S, A, R, SP, O" * "\n")
        @showprogress 1 "Saving History..." for (s, a, r, sp, o, ai, ui) in eachstep(hist, "(s, a, r, sp, o, action_info, update_info)")
            action = get(ACTIONS, a, a)
            write(io, "$s, $action, $r, $sp, $o" * "\n")
            write(io, "Action Info: $ai" * "\n")
            write(io, "Update Info: $ui" * "\n")
        end
    end
end

function save_experiment_data(hist::SimHistory, env::RockSamplePOMDP, i::Int, algo::String)
    output_filename = "results/results_$algo.out"
    if save_gif
        makegif(
            env,
            hist,
            filename="results/gif/$(algo)_$i.gif",
            show_progress=true,
        )
    end
    if save_steps
        save_history(hist, i, algo)
    end

    open(output_filename, "a") do io
        write_and_print(io, "#Steps $algo -> " * string(n_steps(hist)))
        write_and_print(io, "Cumulative Discounted Return $algo -> " * string(discounted_reward(hist)) * "\n")
    end
end

function compute_final_results(data:: Array{Tuple,1}, algo:: String)
    output_filename = "results/results_$algo.out"
    ret = first.(data)
    steps = last.(data)
    open(output_filename, "a") do io
        write_and_print(io, "FINAL RESULTS $algo")
        write_and_print(io, "Avg Retun $(mean(ret))±$(std(ret))")
        write_and_print(io, "Avg Steps $(mean(steps))±$(std(steps))")
    end
end