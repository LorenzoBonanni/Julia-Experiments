step_time = Dict()

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
    a_list = Vector{String}()
    open(output_filename, "w") do io
        write(io, "S, A, R, SP, O" * "\n")
        @showprogress 1 "Saving History..." for (s, a, r, sp, o, ai, ui) in eachstep(hist, "(s, a, r, sp, o, action_info, update_info)")
            action = get(ACTIONS, a, "sense$(a - RockSample.N_BASIC_ACTIONS)")
            push!(a_list, action)
            write(io, "$s, $action, $r, $sp, $o" * "\n")
            write(io, "Action Info: $ai" * "\n")
            write(io, "Update Info: $ui" * "\n")
        end
    end
    # plot_action_dist(a_list, i)
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
    if !haskey(step_time, algo)
        step_time[algo] = []
    end
    append!(step_time[algo], get.(eachstep(hist, "action_info"), :search_time_us, NaN)*1e6)
    open(output_filename, "a") do io
        write_and_print(io, "#Steps $algo -> " * string(n_steps(hist)))
        write_and_print(io, "Cumulative Discounted Return $algo -> " * string(discounted_reward(hist)) * "\n")
    end
end

function compute_final_results(data:: Array{Tuple,1}, algo:: String)
    output_filename = "results/results_$algo.out"
    ret = [x[1] for x in data]
    steps = [x[2] for x in data]
    time = [x[3] for x in data]
    df = DataFrame(disc_return=ret, steps=steps, time=time)
    CSV.write("results/$algo.csv", df)
    open(output_filename, "a") do io
        write_and_print(io, "FINAL RESULTS $algo")
        write_and_print(io, "Avg Retun $(round(mean(ret), digits = 3))±$(round(std(ret), digits = 3))")
        write_and_print(io, "Avg Steps $(round(mean(steps), digits = 3))±$(round(std(steps), digits = 3))")
        write_and_print(io, "Avg Step Time $(round(mean(step_time[algo]), digits = 3))±$(round(std(step_time[algo]), digits=3))")
        write_and_print(io, "Avg Time $(round(mean(time), digits = 3))±$(round(std(time), digits = 3))")
    end
    plot_return(ret, algo)
end