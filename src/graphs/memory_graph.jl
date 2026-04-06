# In-memory graph store implementation

"""
    InMemoryGraphStore <: AbstractGraphStore

A simple in-memory graph store using adjacency lists.
Supports entity embeddings for semantic search, soft-deletion, and multi-tenant filtering.
"""
mutable struct InMemoryGraphStore <: AbstractGraphStore
    llm::AbstractLLM
    embedder::AbstractEmbedder
    threshold::Float64
    custom_prompt::Union{Nothing, String}
    # Storage: node_id => Dict("name", "type", "embedding", "metadata")
    nodes::Dict{String, Dict{String, Any}}
    # Storage: edge_id => Dict("source", "destination", "relationship", "metadata", "valid", ...)
    edges::Dict{String, Dict{String, Any}}
    lock::ReentrantLock
end

function InMemoryGraphStore(; llm::AbstractLLM, embedder::AbstractEmbedder,
                              threshold::Float64=0.7,
                              custom_prompt::Union{Nothing, String}=nothing)
    InMemoryGraphStore(llm, embedder, threshold, custom_prompt,
                       Dict{String, Dict{String, Any}}(),
                       Dict{String, Dict{String, Any}}(),
                       ReentrantLock())
end

# --- Node management ---

function _find_similar_node(graph::InMemoryGraphStore, name::String, filters::Dict;
                             threshold::Union{Nothing, Float64}=nothing)
    th = something(threshold, graph.threshold)
    target_embedding = embed(graph.embedder, name)
    best_id = nothing
    best_score = -Inf

    for (nid, node) in graph.nodes
        # Check filters
        node_meta = get(node, "metadata", Dict{String, Any}())
        filter_match = all(get(node_meta, k, nothing) == v for (k, v) in filters)
        !filter_match && continue

        node_emb = get(node, "embedding", nothing)
        node_emb === nothing && continue

        score = cosine_similarity(target_embedding, Float64.(node_emb))
        if score >= th && score > best_score
            best_score = score
            best_id = nid
        end
    end

    return best_id
end

function _get_or_create_node!(graph::InMemoryGraphStore, name::String, entity_type::String,
                               filters::Dict)::String
    norm_name = normalize_entity(name)

    # Try to find existing similar node
    existing = _find_similar_node(graph, norm_name, filters)
    existing !== nothing && return existing

    # Create new node
    node_id = string(uuid4())
    node_embedding = embed(graph.embedder, norm_name)
    graph.nodes[node_id] = Dict{String, Any}(
        "name" => norm_name,
        "type" => entity_type,
        "embedding" => node_embedding,
        "metadata" => copy(filters),
        "mentions" => 1,
        "created_at" => now_iso(),
    )
    return node_id
end

# Entity/relation extraction is inherited from AbstractGraphStore in base.jl

# --- Core graph operations ---

function add_to_graph!(graph::InMemoryGraphStore, data::String, filters::Dict)
    lock(graph.lock) do
        # Extract entities
        entities = _extract_entities(graph, data)
        user_id = get(filters, "user_id", nothing)

        # Extract relations
        relations = _extract_relations(graph, data; user_id=user_id)

        # Add each relation as nodes + edge
        for rel in relations
            src_type = get(entities, normalize_entity(rel["source"]), "unknown")
            dst_type = get(entities, normalize_entity(rel["destination"]), "unknown")

            src_id = _get_or_create_node!(graph, rel["source"], src_type, filters)
            dst_id = _get_or_create_node!(graph, rel["destination"], dst_type, filters)

            edge_id = string(uuid4())
            graph.edges[edge_id] = Dict{String, Any}(
                "source" => src_id,
                "destination" => dst_id,
                "relationship" => rel["relationship"],
                "metadata" => copy(filters),
                "valid" => true,
                "created_at" => now_iso(),
                "updated_at" => now_iso(),
            )
        end

        return relations
    end
end

function search_graph(graph::InMemoryGraphStore, query::String, filters::Dict; limit::Int=100)
    lock(graph.lock) do
        query_embedding = embed(graph.embedder, normalize_entity(query))
        results = Dict{String, Any}[]

        # Find nodes similar to query
        scored_nodes = Tuple{String, Float64}[]
        for (nid, node) in graph.nodes
            node_meta = get(node, "metadata", Dict{String, Any}())
            filter_match = all(get(node_meta, k, nothing) == v for (k, v) in filters)
            !filter_match && continue

            node_emb = get(node, "embedding", nothing)
            node_emb === nothing && continue
            score = cosine_similarity(query_embedding, Float64.(node_emb))
            score >= graph.threshold && push!(scored_nodes, (nid, score))
        end

        sort!(scored_nodes, by=x -> -x[2])
        matched_node_ids = Set(first.(scored_nodes[1:min(limit, length(scored_nodes))]))

        # Collect edges connected to matched nodes
        for (eid, edge) in graph.edges
            get(edge, "valid", true) || continue
            src_id = edge["source"]
            dst_id = edge["destination"]

            if src_id in matched_node_ids || dst_id in matched_node_ids
                edge_meta = get(edge, "metadata", Dict{String, Any}())
                filter_match = all(get(edge_meta, k, nothing) == v for (k, v) in filters)
                !filter_match && continue

                src_node = get(graph.nodes, src_id, nothing)
                dst_node = get(graph.nodes, dst_id, nothing)
                src_node === nothing && continue
                dst_node === nothing && continue

                push!(results, Dict{String, Any}(
                    "source" => src_node["name"],
                    "relationship" => edge["relationship"],
                    "destination" => dst_node["name"],
                ))
            end
        end

        return results[1:min(limit, length(results))]
    end
end

function delete_from_graph!(graph::InMemoryGraphStore, data::String, filters::Dict)
    lock(graph.lock) do
        for (eid, edge) in graph.edges
            get(edge, "valid", true) || continue
            edge_meta = get(edge, "metadata", Dict{String, Any}())
            filter_match = all(get(edge_meta, k, nothing) == v for (k, v) in filters)
            !filter_match && continue

            # Soft delete
            edge["valid"] = false
            edge["invalidated_at"] = now_iso()
        end
    end
end

function delete_all_graph!(graph::InMemoryGraphStore, filters::Dict)
    lock(graph.lock) do
        # Soft-delete all matching edges
        for (eid, edge) in graph.edges
            edge_meta = get(edge, "metadata", Dict{String, Any}())
            filter_match = all(get(edge_meta, k, nothing) == v for (k, v) in filters)
            if filter_match
                edge["valid"] = false
                edge["invalidated_at"] = now_iso()
            end
        end
    end
end

function get_all_graph(graph::InMemoryGraphStore, filters::Dict; limit::Int=100)
    lock(graph.lock) do
        results = Dict{String, Any}[]
        for (eid, edge) in graph.edges
            get(edge, "valid", true) || continue
            edge_meta = get(edge, "metadata", Dict{String, Any}())
            filter_match = all(get(edge_meta, k, nothing) == v for (k, v) in filters)
            !filter_match && continue

            src_node = get(graph.nodes, edge["source"], nothing)
            dst_node = get(graph.nodes, edge["destination"], nothing)
            src_node === nothing && continue
            dst_node === nothing && continue

            push!(results, Dict{String, Any}(
                "source" => src_node["name"],
                "relationship" => edge["relationship"],
                "destination" => dst_node["name"],
            ))

            length(results) >= limit && break
        end
        return results
    end
end
