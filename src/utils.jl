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
            show_progress=false,
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
        write_and_print(io, "Cumulative Discounted Return $algo -> " * string(discounted_reward(hist)))
        write_and_print(io, "Cumulative Undiscounted Return $algo -> " * string(undiscounted_reward(hist)) * "\n")
    end
end

function compute_final_results(data:: Array{Tuple,1}, algo:: String)
    output_filename = "results/results_$algo.out"
    disc_ret = [x[1] for x in data]
    ret = [x[2] for x in data]
    steps = [x[3] for x in data]
    time = [x[4] for x in data]
    df = DataFrame(disc_return=disc_ret, steps=steps, time=time)
    CSV.write("results/$algo.csv", df)
    open(output_filename, "a") do io
        write_and_print(io, "FINAL RESULTS $algo")
        write_and_print(io, "Avg Disc Retun $(round(mean(disc_ret), digits = 3))±$(round(std(disc_ret), digits = 3))")
        write_and_print(io, "Avg Undisc Retun $(round(mean(ret), digits = 3))±$(round(std(ret), digits = 3))")
        write_and_print(io, "Avg Steps $(round(mean(steps), digits = 3))±$(round(std(steps), digits = 3))")
        write_and_print(io, "Avg Step Time $(round(mean(step_time[algo]), digits = 3))±$(round(std(step_time[algo]), digits=3))")
        write_and_print(io, "Avg Time $(round(mean(time), digits = 3))±$(round(std(time), digits = 3))")
    end
    plot_return(ret, algo)
    plot_return(disc_ret, algo * "disc")
end

function my_simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s)
    
    if sim.eps === nothing
        eps = 0.0
    else
        eps = sim.eps
    end
    
    if sim.max_steps === nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    disc = 1.0
    r_total = 0.0

    b = initialize_belief(updater, initial_belief)

    step = 1

    while disc > eps && !isterminal(pomdp, s) && step <= max_steps

        a = action(policy, s)
        
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, sim.rng)

        r_total += disc*r

        s = sp

        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end

POMDPs.simulate(sim::RolloutSimulator, pomdp::POMDP, policy::Policy, updater::Updater, initial_belief, s) = my_simulate(sim, pomdp, policy, updater, initial_belief, s)