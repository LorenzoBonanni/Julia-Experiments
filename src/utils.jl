

function delete_old_files()
    foreach(rm, filter(endswith(".out"), readdir("results/", join=true)))
    foreach(rm, readdir("results/history", join=true))
    foreach(rm, readdir("results/gif", join=true))
end

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

function compute_final_results(data:: Array{Tuple,1})
    output_filename = "results/results.out"
    ret = first.(data)
    steps = last.(data)
    open(output_filename, "a") do io
        write_and_print(io, "FINAL RESULTS")
        write_and_print(io, "Avg Retun $(mean(ret))±$(std(ret))")
        write_and_print(io, "Avg Steps $(mean(steps))±$(std(steps))")
    end
end