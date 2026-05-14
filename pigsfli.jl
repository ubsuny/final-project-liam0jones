push!(LOAD_PATH, joinpath(dirname(@__FILE__), "src"))
  
using Pigsfli
using JLD2
using TimerOutputs 
using ArgParse
using Printf 
using BenchmarkTools
using Random

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--dimension", "-D"
            help = "Dimension of lattice."
            arg_type = Int
            required = true
            dest_name = "D"
	    "--geometry"
	        help = "Select from square (D-dimensional), triangular/honeycomb/kagome (2D), or custom."
	        arg_type = String
	        required = true
        "--length", "-L"
            help = "Number of unit cells spanned in each dimension (symmetric). Mutually exclusive with --Lx/--Ly/--Lz."
            arg_type = Int
            required = false
            default = nothing
            dest_name = "L"
        "--Lx"
            help = "Number of unit cells along axis 1. Mutually exclusive with --L."
            arg_type = Int
            required = false
            default = nothing
        "--Ly"
            help = "Number of unit cells along axis 2. Mutually exclusive with --L."
            arg_type = Int
            required = false
            default = nothing
        "--Lz"
            help = "Number of unit cells along axis 3 (3D square only). Mutually exclusive with --L."
            arg_type = Int
            required = false
            default = nothing
        "--particle-number", "-N"
            help = "Total number of particles."
            arg_type = Int 
            required = true
            dest_name = "N"
        "--interaction", "-U"
            help = "Interaction potential."
            arg_type = Float64 
            required = true
            dest_name = "U"
        "--subregion-size", "-l"
            help = "Linear size of hypercubic subregion (for calculating subregion-size-resolved local particle number density)."
            arg_type = Int 
            default = 2
            dest_name = "l"
	    "--boundary"
	        help = "Uniform boundary conditions (pbc/obc) for all axes. Mutually exclusive with --bcx/--bcy/--bcz."
	        arg_type = String
	        required = false
	        default = nothing
	        dest_name = "bc"
        "--bcx"
            help = "Boundary condition for axis 1 (pbc/obc). Mutually exclusive with --boundary."
            arg_type = String
            required = false
            default = nothing
        "--bcy"
            help = "Boundary condition for axis 2 (pbc/obc). Mutually exclusive with --boundary."
            arg_type = String
            required = false
            default = nothing
        "--bcz"
            help = "Boundary condition for axis 3 (pbc/obc). Mutually exclusive with --boundary."
            arg_type = String
            required = false
            default = nothing
        "--sweeps"
            help = "Number of sweeps before attempting measurements."
            arg_type = Int 
            default = 10000000
        "--sweeps-pre"
            help = "Number sweeps for each pre-equilibration step."
            arg_type = Int 
            default = 100000
        "--max-blocks"
            help = "Maximum number of pre-equilibration blocks."
            arg_type = Int
            default = 150
        "--beta"
            help = "Set length of imaginary time."
            arg_type = Float64
            required = true
        "--mu"
            help = "Chemical potential."
            arg_type = Float64
            default = -3.0
        "--eta"
            help = "Worm fugacity."
            arg_type = Float64
            default = 1e-3
        "--no-relax-mu"
            help = "Skip mu pre-equilibration and use the initial value."
            action = :store_true 
        "--no-relax-eta"
            help = "Skip eta pre-equilibration and use the initial value."
            action = :store_true 
        "--start-from-state"
            help = "Paths to a state file to start from. Enter with spaces between the filenames: sim_state_path sim_tracker_path [rng_state_path]. The rng_state_path is optional and the rng state will only be reloaded if three paths are given."
            nargs  = '*'
            arg_type = String 
            default = [] 
        "--extra-save-preeq"
            help = "Save the pre-equilibrated state directly after the pre-equilibration phase into separate files. This can be used to later restart different seeds from a single pre-equilibration."
            action = :store_true 
        "--count-total-bins"
            help = "If restarted, count already collected bins and stop when total number of bins reaches nbins (default is false, i.e. collect new nbins bins)."
            action = :store_true 
        "--t", "-t"
            help = "Tunneling parameter."
            arg_type = Float64
            default = 1.0
        "--Z", "-Z"
            help = "Diagonal fraction."
            arg_type = Float64
            default = 48.0
        "--dZ"
            help = "Size of half-window around desired Z."
            arg_type = Float64
            default = 3.0
        "--canonical"
            help = "Set to false for grand canonical simulation."
            arg_type = Bool
            default = true
        "--seed"
            help = "Seed for random number generator."
            arg_type = Int
            default = 0
        "--bin-size"
            help = "Number of measurements per bin."
            arg_type = Int
            default = 10
        "--bins-wanted"
            help = "Number of bins desired in data file."
            arg_type = Int
            default = 1000
        "--subgeometry"
            help = "Shape of subregion: square OR strip."
            arg_type = String
            default = "square"
        "--num-replicas"
            help = "Number of replicas."
            arg_type = Int
            default = 1
        "--measurement-frequency"
            help = "Measurements will be performed every this amount."
            arg_type = Int
            default = 1
        "--measure-tau-resolved-estimators"
            help = "Measure tau resolved estimators."
            action = :store_true 
        "--get-n"
            help = "Compute subregion-size-resolved local particle number density (and squared) for square lattices."
            action = :store_true
	    "--get-density"
	        help = "Get ground state particle number density (and squared)."
	        action = :store_true
	    "--get-corr-mat"
	        help = "Get one-body correlation matrix (and squared)."
	        action = :store_true
        "--get-corr"
            help = "Compute the density-density correlations as corr(r) in 1D square lattices with PBC only."
            action = :store_true 
        "--rng"
            help = "Random Number Generator type ('Xoshiro256pp' or 'MersenneTwister')."
            arg_type = String
            default = "Xoshiro256pp"
        "--trial-state"
            help = "Trial state type. (constant, gutzwiller, non-interacting)"
            arg_type = String
            default = "constant"
        "--kappa"
            help = "Gutzwiller trial state parameter."
            arg_type = Float64
            default = 3.0
        "--restart"
            help = "Continue simulation from a loaded rng state."
            arg_type = Bool
            default = false
        "--no-accessible"
            help = "Do not calculate accessible entanglement entropies." 
            arg_type = Bool
            default = true   
        "--no-flush"
            help = "Do not flush write buffer to output files in after each write."
            action = :store_true
        "--skip-timing"
            help = "Do not record timing information."
            action = :store_true
        "--out"
            help = "Output folder. Defaults to /out/geometry_D/L_N/boundary/U/beta."
            arg_type = String 
            default = "./out"
        "--save-state-every"
            help = "Saving the state is expensive. Only do it every this amount of measurements. For small bin_size, this is important for performance."
            arg_type = Int64
            default = 1
	    "--save-state"
	        help = "Write state, trackers, and RNG data to files."
	        action =:store_true
    end

    args = parse_args(s)

    # ── Restart / state validation ─────────────────────────────────────────────
    if args["start-from-state"] != [] && args["restart"]
        error("Cannot restart (--restart true) if starting state (--start-from-state ...) is given.")
    end

    # ── Geometry validation ────────────────────────────────────────────────────
    geometry = args["geometry"]
    D        = args["D"]
    if !(geometry in ["square","triangular","honeycomb","kagome","custom"])
        error("Unknown geometry '$geometry'. Choose: square, triangular, honeycomb, kagome, or custom.")
    end
    is_2d_geo = geometry in ["triangular","honeycomb","kagome"]
    if is_2d_geo && D != 2
        error("Geometry '$geometry' requires --dimension 2.")
    end

    # ── Per-axis size resolution ───────────────────────────────────────────────
    has_L    = args["L"]  !== nothing
    has_axis = any(!isnothing, [args["Lx"], args["Ly"], args["Lz"]])
    if has_L && has_axis
        error("Cannot specify both --L and --Lx/--Ly/--Lz.")
    end
    if !has_L && !has_axis && geometry != "custom"
        error("Must specify lattice size via --L (symmetric) or --Lx/--Ly/--Lz (per-axis).")
    end
    if has_L && geometry == "custom"
        error("Use --Lx/--Ly/--Lz to specify the span for --geometry custom.")
    end

    n_axes = is_2d_geo ? 2 : (geometry == "custom" ? -1 : D)   # -1 = defer to unit cell

    if geometry != "custom"
        if has_L
            Ls = fill(args["L"], n_axes)
        else
            if n_axes == 1
                args["Lx"] !== nothing || error("--Lx is required for D=1 square lattice.")
                any(!isnothing, [args["Ly"], args["Lz"]]) &&
                    error("--Ly and --Lz are not valid for a D=1 square lattice.")
                Ls = [args["Lx"]]
            elseif n_axes == 2
                args["Lx"] !== nothing || error("--Lx is required for geometry '$geometry'.")
                args["Ly"] !== nothing || error("--Ly is required for geometry '$geometry'.")
                args["Lz"] !== nothing && error("--Lz is not valid for a 2-axis geometry.")
                Ls = [args["Lx"], args["Ly"]]
            else  # n_axes == 3
                args["Lx"] !== nothing || error("--Lx is required for D=3 square lattice.")
                args["Ly"] !== nothing || error("--Ly is required for D=3 square lattice.")
                args["Lz"] !== nothing || error("--Lz is required for D=3 square lattice.")
                Ls = [args["Lx"], args["Ly"], args["Lz"]]
            end
        end
    else
        # For custom, collect all provided axis sizes; final validation after unit cell input.
        Ls_input = Int64.(filter(!isnothing, [args["Lx"], args["Ly"], args["Lz"]]))
        isempty(Ls_input) && error("--geometry custom requires --Lx (and --Ly/--Lz as needed).")
        Ls = Ls_input  # temporary; validated against unit cell dimension below
    end

    # ── Per-axis boundary condition resolution ─────────────────────────────────
    has_bc      = args["bc"]  !== nothing
    has_axis_bc = any(!isnothing, [args["bcx"], args["bcy"], args["bcz"]])
    if has_bc && has_axis_bc
        error("Cannot specify both --boundary and --bcx/--bcy/--bcz.")
    end
    if !has_bc && !has_axis_bc
        error("Must specify boundary conditions via --boundary (uniform) or --bcx/--bcy/--bcz (per-axis).")
    end

    n_bc_axes = length(Ls)   # for custom this is the number of span axes provided

    if has_bc
        bc = args["bc"]
        bc in ["pbc","obc"] || error("--boundary must be pbc or obc.")
        bcs = fill(bc, n_bc_axes)
        bc_label = bc == "pbc" ? "PBC" : "OBC"
    else
        if n_bc_axes == 1
            args["bcx"] !== nothing || error("--bcx is required.")
            bcs = [args["bcx"]]
        elseif n_bc_axes == 2
            args["bcx"] !== nothing || error("--bcx is required.")
            args["bcy"] !== nothing || error("--bcy is required.")
            bcs = [args["bcx"], args["bcy"]]
        else
            args["bcx"] !== nothing || error("--bcx is required.")
            args["bcy"] !== nothing || error("--bcy is required.")
            args["bcz"] !== nothing || error("--bcz is required.")
            bcs = [args["bcx"], args["bcy"], args["bcz"]]
        end
        for bc in bcs
            bc in ["pbc","obc"] || error("Boundary condition '$bc' must be pbc or obc.")
        end
        bc_label = all(==(bcs[1]), bcs) ? (bcs[1]=="pbc" ? "PBC" : "OBC") :
                   join([bc=="pbc" ? "P" : "O" for bc in bcs], "")
    end

    # ── Custom geometry: interactive unit cell input ───────────────────────────
    if geometry == "custom"
        println("\nCustom geometry selected.")
        println("Enter the unit cell translation vectors as a Julia matrix.")
        println("  Rows = spatial dimensions, columns = lattice vectors.")
        println("  Example (triangular): [1.0 0.5; 0.0 0.8660254]")
        print("Translation matrix: ")
        trans_str = readline()
        translations = Float64.(eval(Meta.parse(trans_str)))

        println("Enter basis site coordinates as columns of a Julia matrix,")
        println("  or press Enter for a single site at the origin.")
        println("  Example (honeycomb two-site basis): [0.0 0.3333; 0.0 0.5774]")
        print("Basis sites (or Enter): ")
        basis_str = strip(readline())
        N_dim = size(translations, 1)
        basis_sites = isempty(basis_str) ? zeros(N_dim, 1) : Float64.(eval(Meta.parse(basis_str)))

        args["custom_uc"] = UnitCell(translations, basis_sites)

        length(Ls) == N_dim ||
            error("Custom unit cell has $N_dim axes but $(length(Ls)) span sizes were provided.")
        # Revalidate bcs length against finalized Ls
        length(bcs) == N_dim ||
            error("Custom unit cell has $N_dim axes but $(length(bcs)) boundary conditions were provided.")
    end

    # ── Store normalized sizes and BCs ─────────────────────────────────────────
    args["Ls"]                = Ls
    args["bcs"]               = bcs
    args["L"]                 = Ls[1]   # first axis; used for subsystem calcs and file naming
    args["boundary_condition"] = all(==(bcs[1]), bcs) ? bcs[1] : "mixed"
    args["bc_label"]          = bc_label
    # Human-readable size string: "5" for symmetric, "3x5" for asymmetric
    args["size_str"] = all(==(Ls[1]), Ls) ? string(Ls[1]) : join(string.(Ls), "x")

    # ── Site count M ───────────────────────────────────────────────────────────
    if geometry == "square" || geometry == "triangular"
        args["M"] = prod(Ls)
        z = geometry == "square" ? 2^D : 6
    elseif geometry == "honeycomb"
        args["M"] = 2 * prod(Ls)
        z = 3
    elseif geometry == "kagome"
        args["M"] = 3 * prod(Ls)
        z = 4
    else
        _tmp_lat = span_unitcells(args["custom_uc"], (0:(L-1) for L in Ls)...)
        args["M"] = length(_tmp_lat)
        _tmp_mat = create_adjacency_matrix(Ls, args["D"], geometry, bcs; unitcell=args["custom_uc"])
        z = maximum(_tmp_mat.total_nn)
    end

    # ── Subsystem settings ─────────────────────────────────────────────────────
    if args["get-n"] && geometry != "square"
        error("--get-n is currently only implemented for square lattices.")
    end
    if args["subgeometry"] == "square"
        args["M_A"] = args["l"]^D
    elseif args["subgeometry"] == "strip"
        args["M_A"] = args["l"] * args["L"]
    else
        error("Please choose subgeometry: square OR strip.")
    end
    if !(0 < args["l"] <= args["L"])
        error("Please choose 0 < l <= L (first axis length = $(args["L"])).")
    end

    # ── Simulation sweep counts ────────────────────────────────────────────────
    args["sweep"] = round(Int64, args["beta"] * args["M"])
    MF_Interaction = 2 * z * (2 * args["N"] / args["M"] + 1) * 0.7
    if args["sweeps"] == 10000000
        args["sweeps"] *= round(args["sweep"] * max(2 * sum(args["bcs"].=="obc"), 1) * (args["N"] / args["M"]) * (MF_Interaction / args["U"]), sigdigits = 4)  # Scale default sweeps with projection length, system size, density, and coordination number
    end
    if args["sweeps-pre"] == 100000
        args["sweeps-pre"] *= round(args["sweep"] * max(2 * sum(args["bcs"].=="obc"), 1) * (args["N"] / args["M"]) * (MF_Interaction / args["U"]), sigdigits = 4)
    end

    # ── Measurement settings ───────────────────────────────────────────────────
    args["measurement_center"] = args["beta"] / 2
    args["measurement_plus_minus"] = 0.1 * args["beta"]

    # ── Z / dZ validation ─────────────────────────────────────────────────────
    (args["Z"] < 0 || args["Z"] > 100) && error("Z must be between 0 and 100.")
    ((args["Z"]-args["dZ"]) < 0 || (args["Z"]+args["dZ"]) > 100) &&
        error("Invalid dZ: need Z-dZ > 0 and Z+dZ < 100.")

    # ── Feature-specific validation ────────────────────────────────────────────
    if args["get-corr"] && (args["boundary_condition"] != "pbc" || geometry != "square")
        error("--get-corr is only implemented for square lattices with uniform periodic boundary conditions.")
    end

    return args
end
 

function main()
    # Create a TimerOutput, this is the main type that keeps track of everything.
    to = TimerOutput()
    # print welcome message
    println(pigsfli_str) 
    # parse and print arguments
    args = parse_commandline() 
    println("Run with parameters:")
    let
        skip_if_default = Dict{String,Any}(
            "l" => 2, "subgeometry" => "square", "num-replicas" => 1,
            "measurement-frequency" => 1, "rng" => "Xoshiro256pp",
            "trial-state" => "constant", "kappa" => 3.0,
            "canonical" => true, "seed" => 0,
            "no-accessible" => true, "out" => "./out",
            "save-state-every" => 1, "restart" => false, "L" => nothing, "t" => 1.0, "max-blocks" => 100
        )
        store_true_keys = Set([
            "no-relax-mu", "no-relax-eta", "extra-save-preeq", "count-total-bins",
            "measure-tau-resolved-estimators", "get-n", "get-density",
            "get-corr-mat", "get-corr", "no-flush", "skip-timing", "save-state",
        ])
        for key in [
            "D", "geometry", "Lx", "Ly", "Lz", "M", "N", "U", "l",
            "bc", "bcx", "bcy", "bcz",
            "sweeps", "sweeps-pre", "beta", "mu", "eta",
            "no-relax-mu", "no-relax-eta",
            "start-from-state", "extra-save-preeq", "count-total-bins", "Z", "dZ", "canonical", "seed", "bin-size", "bins-wanted",
            "subgeometry", "num-replicas", "measurement-frequency",
            "measure-tau-resolved-estimators", "get-n", "get-density",
            "get-corr-mat", "get-corr",
            "rng", "trial-state", "kappa", "restart",
            "no-accessible", "no-flush", "skip-timing",
            "out", "save-state-every", "save-state",
        ]
            !haskey(args, key) && continue
            val = args[key]
            val === nothing && continue
            val isa AbstractVector && isempty(val) && continue
            key in store_true_keys && val === false && continue
            haskey(skip_if_default, key) && val == skip_if_default[key] && continue
            println(@sprintf "  %-15s  =>  %s" key val)
        end
        println("")
    end
    # skip timing
    if args["skip-timing"]
        disable_timer!(to)
    end
    # timer 
    @timeit to "pigsfli" begin
      @timeit to "prepare" begin
        # initialize rng
        rng = init_rng(args["rng"], args["seed"])
        # create subsystem 
        sub_sites = create_sub_sites(args["l"], args["L"], args["D"], args["M"], args["subgeometry"])
        # create initial Fock state 
        initial_fock_state = random_boson_config(args["M"], args["N"], rng, args["restart"])
        # create adjacency matrix
	    adjacency_matrix = create_adjacency_matrix(args["Ls"], args["D"], args["geometry"], args["bcs"];
	                                               unitcell=get(args, "custom_uc", nothing))
        # create tracker 
        sim_state = create_simulation_state(args, sub_sites, initial_fock_state, adjacency_matrix)
        sim_options = create_simulation_options(args)
        sim_tracker = create_simulation_tracker(args, sim_state, sim_state.num_replicas)   
      end 
      # setup outputs
      @timeit to "open output files" begin 
        output_fh = create_output_file_handler(sim_state, args)
        state_fh = create_state_file_handler(sim_state, args)
      end
      # Generate JLD2 path
      bc_label = args["bc_label"]
      sz = args["size_str"]
      calculation_label = "$(args["geometry"])_$(args["D"])D_$(sz)L_$(args["N"])N_$(bc_label)_$(round(args["U"],digits=3))U_$(round(args["beta"],digits=2))beta_$(args["bins-wanted"])bins$(args["bin-size"])_seed$(args["seed"])"
      out_folder = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(sz)L_$(args["N"])N/$(bc_label)/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]
      mkpath(out_folder)
      jld2_path = joinpath(out_folder, "$(calculation_label).jld2")
      # pre-equilibration or restart
      if args["restart"]  
        @timeit to "load for restart" begin
            rng, sim_state, sim_tracker =  restart(state_fh)
        end
      elseif args["start-from-state"] != []
        @timeit to "load from state" begin 
            if length(args["start-from-state"]) == 3 
                rng, sim_state, sim_tracker = load_from_state_with_rng(args) 
            elseif length(args["start-from-state"]) == 2
                sim_state, sim_tracker = load_from_state(args) 
            else 
                error("Invalid number of paths given for --start-from-state. Please enter two paths to load state and trackers or three paths to also load rng.")
            end
        end
      else
        @timeit to "pre-equilibration" begin
            if sim_state.canonical 
                pre_equilibrate!(rng, sim_state, sim_options, sim_tracker)
            end
            reset!(sim_state; num_replicas = args["num-replicas"])
            reset!(sim_tracker, sim_state; num_replicas = args["num-replicas"])
            if args["extra-save-preeq"]
                save_state(sim_state, sim_tracker, rng, args; name_prefix="pre-equilibrated")
            end
        end
      end 
      # Monte Carlo
      reset!(sim_tracker.step_counters)     # reset step counters from pre-equilibration
      @timeit to "main MC" begin 
        run_monte_carlo!(rng, sim_state, sim_options, sim_tracker ; output_fh=output_fh, state_fh=state_fh, jld2_path=jld2_path)
      end
      # finalize
      println("")
      print_mc_update_stats(sim_tracker)
    end
    # print timing 
    show(to)
    # close files 
    close(output_fh)
end

"""Loads sim_state, tracker and rng state from files. This is a helper function for loading pre-equilibrated states. Here, we do not use the snapshot handler, as this function is expected to be called only once."""
function load_from_state_with_rng(args)
    path_sim_state, path_sim_tracker, path_rng = args["start-from-state"]

    sim_state = load(path_sim_state, "data")
    sim_tracker = load(path_sim_tracker, "data")
    rng = load(path_rng, "data")
    return rng, sim_state, sim_tracker 
end

function load_from_state(args)
    path_sim_state, path_sim_tracker = args["start-from-state"]

    sim_state = load(path_sim_state, "data")
    sim_tracker = load(path_sim_tracker, "data")
    return sim_state, sim_tracker 
end

"""Saves sim_state, tracker and rng state to files. This function is used to save the final pre-equilibrated states. Here, we do not use the snapshot handler, as this function is expected to be called only once."""
function save_state(sim_state::SimState, sim_tracker::SimTracker, rng::AbstractRNG, args; name_prefix::String="pre-equilibrated")
    bc_label = args["bc_label"]
    sz = args["size_str"]
    calculation_label = "$(args["geometry"])_$(args["D"])D_$(sz)L_$(args["N"])N_$(round(args["U"],digits=3))U_$(bc_label)_$(round(args["beta"],digits=2))beta_$(args["bins-wanted"])bins$(args["bin-size"])_seed$(args["seed"])"
    out_folder = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(sz)L_$(args["N"])N/$(bc_label)/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]

    path_state   = joinpath(out_folder, "system-state_$(name_prefix)_$(calculation_label).dat")
    path_tracker = joinpath(out_folder, "system-trackers_$(name_prefix)_$(calculation_label).dat")
    path_rng     = joinpath(out_folder, "rng_$(name_prefix)_$(calculation_label).dat")

    jldsave(path_state; data=sim_state)
    jldsave(path_tracker; data=sim_tracker)
    jldsave(path_rng; data=rng)

    return nothing
end

"""Define and open all output files. Defines how data is written to file. For a restarted simulation, append to files."""
function create_output_file_handler(sim_state, args)::WriteOutputHandler
    output_fh = WriteOutputHandler(~args["no-flush"])
    bc_label = args["bc_label"]
    sz = args["size_str"]
    calculation_label = "$(args["geometry"])_$(args["D"])D_$(sz)L_$(args["N"])N_$(bc_label)_$(round(args["U"],digits=3))U_$(round(args["beta"],digits=2))beta_$(args["bins-wanted"])bins$(args["bin-size"])_seed$(args["seed"])"
    open_mode = args["restart"] ? "a" : "w"
    out_folder = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(sz)L_$(args["N"])N/$(bc_label)/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]
    mkpath(out_folder)
    ##### conventional estimator files ##### 
    if args["num-replicas"] < 2 
        # tau resolved measurements 
        if args["measure-tau-resolved-estimators"]
            # tau resolved kinetic energy: K(tau)
                handler_name = "tr_kinetic_energy"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "tauResolvedK_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name) 
            # tau resolved potential energy: V(tau)
                handler_name = "tr_diagonal_energy"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "tauResolvedV_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name) 
        end
        # Local particle number distribution (and squared)
        if args["get-n"] 
            # local particle number distribution: <n>
                handler_name = "n"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data[1]], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "n_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name) 
            # local particle number squared distribution: <n^2>
                handler_name = "n_squared"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data[1]], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "n_squared_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name) 
        end

	    # Correlations in fluctuations 
        if args["get-corr"]
            # correlation function
                handler_name = "corr"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data[1]], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "corr_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name)  
            # fluctuations
                handler_name = "sigma2"
                # function to convert data to string, data = K
                out_str = (data) -> @sprintf "%s\n" join([@sprintf "%.20f" d for d in data[1]], " ") 
                # open file
                path = joinpath(out_folder, @sprintf "sigma2_%s.dat" calculation_label)
                file = open(path, open_mode)
                # add to file_handler
                add!(output_fh, file, out_str, handler_name)            
        end
    end
    ##### estimators in replicated configuration space ##### 
    if args["num-replicas"] >= 2 

        if args["canonical"]
            # canonical
                # swap histogram 
                    handler_name = "SWAP_histogram"
                    # function to convert data to string, data = K
                    out_str = (data) ->  @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ")
                    # open file
                    path = joinpath(out_folder, @sprintf "SWAP_%s.dat" calculation_label)
                    file = open(path, open_mode)
                    # add to file_handler
                    add!(output_fh, file, out_str, handler_name) 
                if ~args["no-accessible"]
                    # SWAP histograms for each partition size mA
                    for mA in 1:args["M_A"]
                        handler_name = @sprintf "SWAPn-mA%d" mA
                        # function to convert data to string, data = K
                        out_str = (data) -> @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ") 
                        # open file
                        path = joinpath(out_folder, @sprintf "SWAPn-mA%d_%s.dat" mA calculation_label)
                        file = open(path, open_mode)
                        # add to file_handler
                        add!(output_fh, file, out_str, handler_name) 
                    end
                    # P(n) for each partition size mA
                    for mA in 1:args["M_A"]
                        handler_name = @sprintf "Pn-mA%d" mA
                        # function to convert data to string, data = K
                        out_str = (data) -> @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ") 
                        # open file
                        path = joinpath(out_folder, @sprintf "Pn-mA%d_%s.dat" mA calculation_label)
                        file = open(path, open_mode)
                        # add to file_handler
                        add!(output_fh, file, out_str, handler_name) 
                    end
                    # P(n)^2? for each partition size mA
                    for mA in 1:args["M_A"]
                        handler_name = @sprintf "PnSquared-mA%d" mA
                        # function to convert data to string, data = K
                        out_str = (data) -> @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ") 
                        # open file
                        path = joinpath(out_folder, @sprintf "PnSquared-mA%d_%s.dat" mA calculation_label)
                        file = open(path, open_mode)
                        # add to file_handler
                        add!(output_fh, file, out_str, handler_name) 
                    end
                end
        else 
            # grand canonical
                # swap histogram 
                    handler_name = "SWAP_grandcan_histogram"
                    # function to convert data to string, data = K
                    out_str = (data) -> @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ") 
                    # open file
                    path = joinpath(out_folder, @sprintf "SWAP_grandcan_%s.dat" calculation_label)
                    file = open(path, open_mode)
                    # add to file_handler
                    add!(output_fh, file, out_str, handler_name)

                if ~args["no-accessible"]
                    # SWAP histograms for each n-Sector 
                    for mA in 0:args["N"]
                        handler_name = @sprintf "SWAPn_grandcan-%d-sector" mA
                        # function to convert data to string, data = K
                        out_str = (data) -> @sprintf "%s\n" join([@sprintf "%d" d for d in data[1]], " ") 
                        # open file
                        path = joinpath(out_folder, @sprintf "SWAPn_grandcan-%d-sector_%s.dat" mA calculation_label)
                        file = open(path, open_mode)
                        # add to file_handler
                        add!(output_fh, file, out_str, handler_name) 
                    end
                end
        end

    end

    return output_fh
end

function get_missing_number_of_bins(bins_wanted, args)
    bc_label = args["bc_label"]
    sz = args["size_str"]
    out_folder = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(sz)L_$(args["N"])N/$(bc_label)/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]
    calculation_label = "$(args["geometry"])_$(args["D"])D_$(sz)L_$(args["N"])N_$(bc_label)_$(round(args["U"],digits=3))U_$(round(args["beta"],digits=2))beta_$(args["bins-wanted"])bins$(args["bin-size"])_seed$(args["seed"])"

    path = args["num-replicas"] < 2 ?
           joinpath(out_folder, "K_$(calculation_label).dat") :
           joinpath(out_folder, "SWAP_$(calculation_label).dat")

    n_bins = 0
    open(path, "r") do f
        n_bins = length(readlines(f))
    end
    return bins_wanted - n_bins
end

function create_state_file_handler(sim_state, args) 

    state_fh = SnapshotHandler()
    bc_label = args["bc_label"]
    sz = args["size_str"]
    calculation_label = "$(args["geometry"])_$(args["D"])D_$(sz)L_$(args["N"])N_$(bc_label)_$(round(args["U"],digits=3))U_$(round(args["beta"],digits=2))beta_$(args["bins-wanted"])bins$(args["bin-size"])_seed$(args["seed"])"
    out_folder = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(sz)L_$(args["N"])N/$(bc_label)/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]
    mkpath(out_folder)
    updata_every_steps = args["save-state-every"]

    handler_name = "state"
    path = joinpath(out_folder, "system-state_$(calculation_label).dat")

    add!(state_fh, path, updata_every_steps, handler_name)

    handler_name = "tracker"
    path = joinpath(out_folder, "system-trackers_$(calculation_label).dat")

    add!(state_fh, path, updata_every_steps, handler_name)

    # rng  
    handler_name = "rng"
    path = joinpath(out_folder, "rng_$(calculation_label).dat")

    add!(state_fh, path, updata_every_steps, handler_name)

    return state_fh
end

function create_simulation_state(
    args,  
    sub_sites::Sites, 
    initial_fock_state::FockState, 
    adjacency_matrix::Adjacency_Matrix)
    
    L::Int64 = args["L"]
    N::Int64 = args["N"]
    M::Int64 = args["M"]
    D::Int64 = args["D"]
    m_A::Int64 = args["M_A"]
    t::Float64 = args["t"]
    U::Float64 = args["U"]
    beta::Float64 = args["beta"]
    num_replicas::Int64 = args["num-replicas"]
    canonical::Bool = args["canonical"]
    eta::Float64 = args["eta"]
    mu::Float64 = args["mu"] 
    rng_type::String = args["rng"]
    seed::Int64 = args["seed"]
    trial_state::TrialState = create_trial_state(args)
    model::Type = BoseHubbard

    return SimState( 
        initial_fock_state,
        adjacency_matrix,
        sub_sites,
        L,
        N,
        M,
        D,
        m_A,
        t,
        U,
        beta,
        num_replicas,
        canonical,
        mu,
        eta,
        rng_type,
        seed,
        trial_state,
        model
    ) 
end  

function create_trial_state(args)
    if args["trial-state"] == "constant"
        return ConstantTrialState()
    elseif args["trial-state"] == "gutzwiller"
        return GutzwillerTrialState(args["kappa"])
    elseif args["trial-state"] == "non-interacting"
        return NonInteractingTrialState()
    end 

    error("Please choose trial state: 'constant' OR 'gutzwiller' OR 'non-interacting'.")
end

function create_simulation_options(args)
    restart::Int64 = args["restart"] 
    num_replicas::Int64 = args["num-replicas"] 
    sweeps_pre::Int64 = args["sweeps-pre"]
    sweeps::Int64 = args["sweeps"]
    sweep::Int64 = args["sweep"]
    max_blocks::Int64 = args["max-blocks"]
    measurement_frequency::Int64 = args["measurement-frequency"]
    measure_tau_resolved_estimators::Bool = args["measure-tau-resolved-estimators"]
    measure_n::Bool = args["get-n"]
    measure_density::Bool = args["get-density"]
    measure_corr_mat::Bool = args["get-corr-mat"]
    measure_corr::Bool = args["get-corr"]
    no_accessible::Bool = args["no-accessible"]
    bins_wanted::Int64 = args["bins-wanted"]
    bin_size::Int64 = args["bin-size"]
    Z::Float64 = args["Z"]/100.0
    dZ::Float64 = args["dZ"]/100.0
    norelax_mu_preeq::Bool = args["no-relax-mu"]
    norelax_eta_preeq::Bool = args["no-relax-eta"]
    save_state::Bool = args["save-state"]
    out_folder::String = args["out"]=="./out" ? "$(args["out"])/$(args["geometry"])_$(args["D"])D/$(args["size_str"])L_$(args["N"])N/$(args["bc_label"])/$(round(args["U"],digits=4))U/$(round(args["beta"],digits=4))beta" : args["out"]
 
    if args["count-total-bins"]
        bins_wanted = get_missing_number_of_bins(bins_wanted, args)
        println("Found already collected bins in restarted file. Only require $(bins_wanted) more bins.")
    end

    return SimOptions(
        restart, 
        num_replicas,
        sweeps_pre,
        sweeps,
        sweep,
        max_blocks,
        measurement_frequency,
        measure_tau_resolved_estimators,
        measure_n,
	    measure_density,
	    measure_corr_mat,
        measure_corr,
        no_accessible,
        bins_wanted,
        bin_size,
        Z,
        dZ,
        norelax_mu_preeq,
        norelax_eta_preeq,
	    save_state,
        out_folder)
end

function create_simulation_tracker(args, sim_state::SimState, num_replicas::Int64) 
    return SimTracker(sim_state; num_replicas=num_replicas)  
end 
 
if abspath(PROGRAM_FILE) == @__FILE__
    #### run simulation ####
    main() 
end
