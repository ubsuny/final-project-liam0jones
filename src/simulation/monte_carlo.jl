using ProgressBars

const CHECK_CONFIG::Bool = false

function run_monte_carlo!(
    rng::AbstractRNG,
    sim_state::SimState,
    sim_options::SimOptions,
    sim_tracker::SimTracker;
    output_fh::WriteOutputHandler,
    state_fh::SnapshotHandler,
    jld2_path::String=""
)
    sim_tracker.sim_phase = monte_carlo_phase
    num_replicas::Int64 = sim_state.num_replicas
    sweeps::Int64 = sim_options.sweeps
    sweeps_between_measurement::Int64 = sim_options.sweep * sim_options.measurement_frequency
    perform_swap_update::Bool = (num_replicas > 1)
    bins_wanted::Int64 = sim_options.bins_wanted
    bin_size::Int64 = sim_options.bin_size

    stage2_corr_bin_count::Int64 = 0
    if !sim_options.restart
        println("Stage (2/3): Equilibrating...\n")
        eq_sweep_interval::Int64 = sweeps_between_measurement
        n_prev::Vector{Float64} = fill(sim_state.N / sim_state.M, sim_state.M)
        n_current::Vector{Float64} = zeros(Float64, sim_state.M)
        density_accum::Vector{Float64} = zeros(Float64, sim_state.M)
        density_sq_accum::Vector{Float64} = zeros(Float64, sim_state.M)
        eq_conv_size::Int64 = max(bin_size, 100)
        density_ctr::Int64 = 0
        sweeps_done::Int64 = 0
        pct_dif::Float64 = 100.0
        converge_ctr::Int64 = 0
        eq_ready::Bool = false
        stage2_corr_ctr::Int64 = 0

        run_sweeps!(sim_state, sim_tracker, rng, sweeps, num_replicas, perform_swap_update)
        sweeps_done += sweeps
        println("  Equilibration sweeps completed. Starting density convergence tracking...")
        jld2_fh = (sim_options.measure_corr_mat && jld2_path != "") ? jldopen(jld2_path, "a+") : nothing
        try
            while !eq_ready || (sim_options.measure_corr_mat && stage2_corr_bin_count < bins_wanted)
                run_sweeps!(sim_state, sim_tracker, rng, eq_sweep_interval, num_replicas, perform_swap_update)
                sweeps_done += eq_sweep_interval

                # Density convergence tracking
                if !eq_ready
                    if sim_state.head_idx[1] == -1 && sim_state.tail_idx[1] == -1 && sim_tracker.N_zero[1] == sim_state.N && sim_tracker.N_beta[1] == sim_state.N
                        update_fock_state!(sim_state, 1, sim_tracker)
                        density_accum .+= sim_state.fock_state_at_slice.data
                        density_sq_accum .+= Float64.(sim_state.fock_state_at_slice.data) .^ 2
                        density_ctr += 1
                        if density_ctr >= eq_conv_size
                            n_current = density_accum ./ density_ctr
                            n_sq_mean = density_sq_accum ./ density_ctr
                            n_pooled = (n_current .+ n_prev) ./ 2
                            site_var = max.(n_sq_mean .- n_current .^ 2, n_pooled)
                            site_sem = sqrt.(site_var ./ density_ctr)
                            pct_dif = sum(abs.(n_current .- n_prev) ./ (site_sem .+ 1e-6)) / sim_state.M
                            if pct_dif <= 1.0
                                converge_ctr += 1
                                if converge_ctr >= 10
                                    eq_ready = true
                                    if sim_options.measure_corr_mat
                                        eq_sweep_interval = maximum([eq_sweep_interval ÷ 2, sim_options.measurement_frequency])
                                        println("  Density converged with sem_dif = $(round(pct_dif, sigdigits=3)) SEM. Starting corr_mat collection at sweep $(sweeps_done).")
                                    else
                                        println("  Density converged with sem_dif = $(round(pct_dif, sigdigits=3)) SEM at sweep $(sweeps_done).")
                                    end
                                end
                            else
                                converge_ctr = 0
                            end
                            n_prev .= n_current
                            fill!(density_accum, 0.0)
                            fill!(density_sq_accum, 0.0)
                            density_ctr = 0
                        end
                    end
                end

                # corr_mat collection (only after density convergence)
                if eq_ready && sim_options.measure_corr_mat && stage2_corr_bin_count < bins_wanted
                    if sim_state.head_idx[1] != -1 && sim_state.tail_idx[1] != -1
                        stage2_corr_ctr = measure_corr_mat!(sim_state, sim_tracker, 1, stage2_corr_ctr)
                    end
                    if stage2_corr_ctr >= bin_size
                        stage2_corr_bin_count += 1
                        if jld2_fh !== nothing
                            C_ij_buf = sim_tracker.corr_mat ./ stage2_corr_ctr
                            C_ij_buf .= (C_ij_buf .+ C_ij_buf') ./ 2
                            jld2_fh[@sprintf("C_%04d", stage2_corr_bin_count)] = C_ij_buf
                        end
                        fill!(sim_tracker.corr_mat, 0.0)
                        stage2_corr_ctr = 0
                    end
                end
            end
        finally
            jld2_fh !== nothing && close(jld2_fh)
        end
        if sim_options.measure_corr_mat
            if stage2_corr_bin_count < bins_wanted
                println("  Warning: only $(stage2_corr_bin_count) of $(bins_wanted) corr_mat bins collected in Stage 2.\n")
            else
                println("  All $(bins_wanted) corr_mat bins collected in Stage 2.\n")
            end
        end
    else
        println("Stage (2/3): RESTARTED SIMULATION: Equilibration not needed\n")
    end

    println("Stage (3/3): Main Monte Carlo loop...")
    for m_count = tqdm(1:bins_wanted)
        writing_ctr::Int64 = 0
        while writing_ctr < bin_size
            run_sweeps!(sim_state, sim_tracker, rng, sweeps_between_measurement, num_replicas, perform_swap_update)
            if CHECK_CONFIG && !is_configuration_valid(sim_state)
                error("Invalid configuration found!")
            end
            if !perform_swap_update
                writing_ctr = conventional_measurement!(sim_tracker, sim_state, sim_options, writing_ctr)
            else
                writing_ctr = swap_measurement!(sim_tracker, sim_state, sim_options, writing_ctr)
            end
        end
        if !perform_swap_update
            write_to_file_conventional!(sim_tracker, output_fh, sim_options, sim_state, jld2_path, m_count; corr_mat_bin_count=0, corr_mat_sample_count=0)
            reset_conventional_measurement!(sim_tracker, sim_options)
        else
            write_to_file_swap(sim_tracker, output_fh, sim_options, sim_state)
            reset_swap_measurement!(sim_tracker, sim_options)
        end
        if sim_options.save_state
            write_state_to_file(state_fh, sim_state, sim_tracker, rng, m_count)
        end
    end
    if sim_options.save_state
        write_state_to_file(state_fh, sim_state, sim_tracker, rng, -1)
    end
end

function run_sweeps!(sim_state::SimState, sim_tracker::SimTracker, rng::AbstractRNG, n_sweeps::Int64, num_replicas::Int64, perform_swap_update::Bool)
    for _ = 1:n_sweeps
        # ---- single replica update ----  
        for r=1:num_replicas
            # Run a single update on replica r 
            random_mc_update!(rng, sim_state, r, sim_tracker)
        end
        # ---- swap update ----
        if perform_swap_update
            random_swap_update!(rng, sim_state, 1, sim_tracker)
        end
    end
end
  
function write_to_file_conventional!(sim_tracker::SimTracker, output_fh::WriteOutputHandler, sim_options::SimOptions, sim_state::SimState, jld2_path::String="", bin_count::Int64=0; corr_mat_bin_count::Int64=bin_count, corr_mat_sample_count::Int64=0)
    # Round out N_tracker since it might have floating point errors after a while
    sim_tracker.Ns = round.(sim_tracker.Ns)
    # Conventional measurements
    jldopen(jld2_path != "" ? jld2_path : "/dev/null", "a+") do file
        file["K_$(bin_count)"] = sim_tracker.kinetic_energy ./ sim_options.bin_size
        file["V_$(bin_count)"] = sim_tracker.diagonal_energy ./ sim_options.bin_size
    end

    if sim_options.measure_tau_resolved_estimators
        write(output_fh, "tr_kinetic_energy", sim_tracker.tr_kinetic_energy[1] ./ sim_options.bin_size)
        write(output_fh, "tr_diagonal_energy", sim_tracker.tr_diagonal_energy[1] ./ sim_options.bin_size) 
    end 
    # write <n> and <n^2> to disk
    if sim_options.measure_n
        write(output_fh, "n", sim_tracker.n_A_accum ./ sim_options.bin_size)
        write(output_fh, "n_squared", sim_tracker.n_A_squared_accum ./ sim_options.bin_size)
    end

    # Write <n_i> and <n^2_i> to disk
    if sim_options.measure_density
        jldopen(jld2_path != "" ? jld2_path : "/dev/null", "a+") do file
            file["n_$(bin_count)"] = sim_tracker.density ./ sim_options.bin_size
            file["n^2_$(bin_count)"] = sim_tracker.density_squared ./ sim_options.bin_size
        end
    end 

    # Write C_ij to JLD2 file (one dataset per bin for jackknife analysis).
    # This ensures consistent number of bins for all observables.
    corr_mat_bin_written::Bool = false
    if sim_options.measure_corr_mat && jld2_path != "" && corr_mat_bin_count > 0 && corr_mat_sample_count >= sim_options.bin_size
        C_ij = sim_tracker.corr_mat ./ corr_mat_sample_count
        C_ij .= (C_ij .+ C_ij') ./ 2
        jldopen(jld2_path, "a+") do file
            file["C_$(corr_mat_bin_count)"] = C_ij
        end
        corr_mat_bin_written = true
        fill!(sim_tracker.corr_mat, 0.0)
    end

    # write C and sigma2 to disk
    if sim_options.measure_corr
        write(output_fh, "corr", sim_tracker.corr_accum ./ sim_options.bin_size) 
        
        rho02::Float64 = (sim_state.N/sim_state.L)^2
        for i = 0:length(sim_tracker.sigma2_accum)-1
            sim_tracker.sigma2_accum[i+begin] = sim_tracker.sigma2_accum[i+begin] ./ sim_options.bin_size - rho02 * (i+1) * (i+1)
        end
        write(output_fh, "sigma2", sim_tracker.sigma2_accum)
    end

    return corr_mat_bin_written
end
 
function write_to_file_swap(sim_tracker::SimTracker, output_fh::WriteOutputHandler, sim_options::SimOptions, sim_state::SimState)
    # Write SWAP histogram
    write(output_fh, "SWAP_histogram", sim_tracker.SWAP_histogram)  
                    
    if !sim_options.no_accessible
        for mA = 1:sim_state.m_A
            # Write Pn 
            write(output_fh, (@sprintf "Pn-mA%d" mA), sim_tracker.Pn[mA])
            # Write Pn_squared
            write(output_fh, (@sprintf "PnSquared-mA%d" mA), sim_tracker.Pn_squared[mA])
            # Write SWAPn_histograms
            write(output_fh, (@sprintf "SWAPn-mA%d" mA), sim_tracker.SWAPn_histograms[mA])
        end
    end
end

function write_state_to_file(state_fh::SnapshotHandler, sim_state::SimState, sim_tracker::SimTracker, rng::AbstractRNG, m_count::Int64)
    if time_for_snapshot(state_fh, "state", m_count)
        # write state to file and reset counter (after any measurement)  
        write(state_fh, "state", sim_state)
        # write out trackers 
        write(state_fh, "tracker", sim_tracker)
        # write out rng 
        write(state_fh, "rng", rng) 
    end
end
