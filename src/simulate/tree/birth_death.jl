function _extant_count_by_node(children::Vector{Vector{Int32}}, active::Vector{Int32})
    nnodes = length(children)
    counts = zeros(Int32, nnodes)
    for tip in active
        counts[Int(tip)] = 1
    end
    for node in nnodes:-1:1
        total = counts[node]
        for child in children[node]
            total += counts[Int(child)]
        end
        counts[node] = total
    end
    return counts
end

function _prune_extant_simtree(
    parent::Vector{Int32},
    children::Vector{Vector{Int32}},
    node_time::Vector{Float64},
    active::Vector{Int32};
    tip_prefix::AbstractString,
)
    counts = _extant_count_by_node(children, active)
    ntips = length(active)
    root = Int32(1)
    while true
        extant_children = Int32[]
        for child in children[Int(root)]
            counts[Int(child)] > 0 && push!(extant_children, child)
        end
        length(extant_children) == 1 || break
        root = only(extant_children)
    end

    new_parent = Int32[]
    new_children = Vector{Vector{Int32}}()
    new_time = Float64[]
    new_is_tip = BitVector()

    function add_node(old::Int32, newpar::Int32)
        new_id = Int32(length(new_parent) + 1)
        push!(new_parent, newpar)
        push!(new_children, Int32[])
        push!(new_time, node_time[Int(old)] - node_time[Int(root)])
        push!(new_is_tip, false)
        newpar != 0 && push!(new_children[Int(newpar)], new_id)
        return new_id
    end

    root_new = add_node(root, Int32(0))
    stack = Tuple{Int32, Int32}[(root, root_new)]
    active_set = Set(active)
    while !isempty(stack)
        old, newpar = pop!(stack)
        extant_children = Int32[]
        for child in children[Int(old)]
            counts[Int(child)] > 0 && push!(extant_children, child)
        end
        if isempty(extant_children)
            new_is_tip[Int(newpar)] = true
            continue
        end
        for child in Iterators.reverse(extant_children)
            current = child
            while !(current in active_set)
                nexts = Int32[]
                for grandchild in children[Int(current)]
                    counts[Int(grandchild)] > 0 && push!(nexts, grandchild)
                end
                length(nexts) == 1 || break
                current = only(nexts)
            end
            new_child = add_node(current, newpar)
            push!(stack, (current, new_child))
        end
    end

    tip_labels = ["$(tip_prefix)$(i)" for i in 1:ntips]
    return SimulatedTree(
        parent = new_parent,
        children = new_children,
        node_time = new_time,
        is_tip = new_is_tip,
        tip_labels = tip_labels,
    )
end

function _simulate_birth_death_attempt(
    n_tips::Integer;
    birth_rate::Real,
    death_rate::Real,
    tree_height::Union{Nothing, Real},
    tip_prefix::AbstractString,
    rng::AbstractRNG,
)
    max_nodes = max(2 * Int(n_tips) - 1, 16)
    parent = zeros(Int32, max_nodes)
    children = [Int32[] for _ in 1:max_nodes]
    node_time = zeros(Float64, max_nodes)
    active = Int32[1]
    next_node = 2
    current_time = 0.0
    birth_prob = Float64(birth_rate) / (Float64(birth_rate) + Float64(death_rate))

    while !isempty(active) && length(active) < n_tips
        current_time += randexp(rng) / ((Float64(birth_rate) + Float64(death_rate)) * length(active))
        idx = rand(rng, 1:length(active))
        lineage = active[idx]
        if rand(rng) < birth_prob
            if next_node + 1 > length(parent)
                old_len = length(parent)
                new_len = max(next_node + 1, 2 * old_len)
                resize!(parent, new_len)
                fill!(view(parent, old_len + 1:new_len), Int32(0))
                append!(children, (Int32[] for _ in old_len + 1:new_len))
                resize!(node_time, new_len)
                fill!(view(node_time, old_len + 1:new_len), 0.0)
            end
            left = Int32(next_node)
            right = Int32(next_node + 1)
            next_node += 2
            node_time[Int(left)] = current_time
            node_time[Int(right)] = current_time
            parent[Int(left)] = lineage
            parent[Int(right)] = lineage
            push!(children[Int(lineage)], left)
            push!(children[Int(lineage)], right)
            active[idx] = left
            push!(active, right)
        else
            active[idx] = active[end]
            pop!(active)
        end
    end
    isempty(active) && return nothing

    used = next_node - 1
    resize!(parent, used)
    resize!(children, used)
    resize!(node_time, used)
    for tip in active
        node_time[Int(tip)] = current_time
    end
    simtree = _prune_extant_simtree(parent, children, node_time, active; tip_prefix = tip_prefix)
    if tree_height !== nothing
        tip_height = maximum(simtree.node_time[Int(tip)] for tip in findall(simtree.is_tip))
        scale = Float64(tree_height) / tip_height
        simtree.node_time .*= scale
    end
    return simtree
end

function simulate_birth_death_simtree(
    n_tips::Integer;
    birth_rate::Real,
    death_rate::Real,
    tree_height::Union{Nothing, Real} = nothing,
    tip_prefix::AbstractString = "t",
    max_attempts::Integer = 1000,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    n_tips >= 2 || throw(ArgumentError("simulate_birth_death_tree requires at least 2 tips"))
    birth_rate > 0 || throw(ArgumentError("birth_rate must be positive"))
    death_rate >= 0 || throw(ArgumentError("death_rate must be non-negative"))
    tree_height === nothing || tree_height > 0 || throw(ArgumentError("tree_height must be positive"))
    max_attempts >= 1 || throw(ArgumentError("max_attempts must be positive"))
    death_rate == 0 && return simulate_yule_simtree(
        n_tips;
        birth_rate = birth_rate,
        tree_height = tree_height,
        tip_prefix = tip_prefix,
        rng = rng,
    )

    for _ in 1:max_attempts
        simtree = _simulate_birth_death_attempt(
            n_tips;
            birth_rate = birth_rate,
            death_rate = death_rate,
            tree_height = tree_height,
            tip_prefix = tip_prefix,
            rng = rng,
        )
        simtree !== nothing && return simtree
    end
    throw(ArgumentError("birth-death simulation went extinct in all max_attempts attempts"))
end

function simulate_birth_death_tree(args...; kwargs...)
    return serialize_tree(simulate_birth_death_simtree(args...; kwargs...))
end
