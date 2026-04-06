# Neo4j graph store implementation using Neo4j HTTP Transactional API

using Base64

"""
    Neo4jGraphStore <: AbstractGraphStore

Graph store backed by Neo4j, accessed via the HTTP Transactional Cypher API.
Stores entity embeddings as float-array node properties for client-side
cosine similarity matching, and uses Cypher MERGE for upsert semantics.

# Configuration

Pass a config dict with keys:
- `"url"` — Neo4j HTTP API URL (default `"http://localhost:7474"`)
  Also accepts bolt/neo4j URLs (`neo4j://host:port`) — the host is extracted
  and HTTP port 7474 is used.
- `"username"` — required
- `"password"` — required
- `"database"` — database name (default `"neo4j"`)
"""
mutable struct Neo4jGraphStore <: AbstractGraphStore
    llm::AbstractLLM
    embedder::AbstractEmbedder
    threshold::Float64
    custom_prompt::Union{Nothing, String}
    http_url::String
    database::String
    auth_header::String
end

function Neo4jGraphStore(; llm::AbstractLLM, embedder::AbstractEmbedder,
                           config::Dict{String, Any}=Dict{String, Any}(),
                           threshold::Float64=0.7,
                           custom_prompt::Union{Nothing, String}=nothing)
    username = get(config, "username", nothing)
    password = get(config, "password", nothing)
    username === nothing && throw(Mem0Error("Neo4j config requires 'username'"))
    password === nothing && throw(Mem0Error("Neo4j config requires 'password'"))

    raw_url = get(config, "url", "http://localhost:7474")
    http_url = _resolve_neo4j_http_url(raw_url)
    database = get(config, "database", "neo4j")
    auth = base64encode("$(username):$(password)")

    return Neo4jGraphStore(llm, embedder, threshold, custom_prompt,
                           http_url, database, "Basic $auth")
end

"""Resolve a bolt/neo4j URL to the HTTP API base URL."""
function _resolve_neo4j_http_url(url::String)::String
    if startswith(url, "http://") || startswith(url, "https://")
        return rstrip(url, '/')
    end
    # Extract host from bolt:// or neo4j:// URLs
    m = match(r"(?:bolt|neo4j)(?:\+s)?://([^:/]+)", url)
    host = m !== nothing ? m.captures[1] : "localhost"
    return "http://$(host):7474"
end

# ── Cypher HTTP API helpers ──────────────────────────────────────────

"""Execute one or more Cypher statements and return the results array."""
function _cypher(graph::Neo4jGraphStore, statements::Vector{<:Dict})
    url = "$(graph.http_url)/db/$(graph.database)/tx/commit"
    body = JSON3.write(Dict("statements" => statements))
    resp = HTTP.post(url,
        ["Content-Type" => "application/json", "Authorization" => graph.auth_header],
        body; status_exception=false)

    resp.status in 200:299 || throw(Mem0Error(
        "Neo4j HTTP error $(resp.status): $(String(resp.body))"))

    parsed = JSON3.read(String(resp.body), Dict{String, Any})
    errors = get(parsed, "errors", [])
    if !isempty(errors)
        msg = join([get(e, "message", string(e)) for e in errors], "; ")
        throw(Mem0Error("Neo4j Cypher error: $msg"))
    end
    return get(parsed, "results", [])
end

"""Shorthand: run a single Cypher statement with parameters."""
function _cypher_query(graph::Neo4jGraphStore, query::String,
                       params::Dict{String, Any}=Dict{String, Any}())
    stmt = Dict{String, Any}("statement" => query, "parameters" => params)
    results = _cypher(graph, [stmt])
    isempty(results) && return (columns=String[], rows=Vector{Any}[])
    r = results[1]
    cols = get(r, "columns", String[])
    data = get(r, "data", [])
    rows = [get(d, "row", []) for d in data]
    return (columns=cols, rows=rows)
end

# ── Filter helpers ───────────────────────────────────────────────────

"""Build a Cypher WHERE clause fragment and params dict for filters."""
function _filter_clause(filters::Dict, var::String="n";
                        prefix::String="f")::Tuple{String, Dict{String, Any}}
    parts = String[]
    params = Dict{String, Any}()
    for (i, (k, v)) in enumerate(filters)
        pname = "$(prefix)_$(k)"
        push!(parts, "$(var).$(k) = \$$(pname)")
        params[pname] = v
    end
    clause = isempty(parts) ? "" : " AND " * join(parts, " AND ")
    return (clause, params)
end

"""Build a SET clause fragment to stamp filters on a node."""
function _filter_set_clause(filters::Dict, var::String="n")::String
    parts = String[]
    for (k, _) in filters
        push!(parts, "$(var).$(k) = \$f_$(k)")
    end
    isempty(parts) && return ""
    return ", " * join(parts, ", ")
end

function _filter_params(filters::Dict)::Dict{String, Any}
    Dict{String, Any}("f_$k" => v for (k, v) in filters)
end

# ── Node management ──────────────────────────────────────────────────

function _find_similar_node(graph::Neo4jGraphStore, name::String, filters::Dict;
                             threshold::Union{Nothing, Float64}=nothing)
    th = something(threshold, graph.threshold)
    target_embedding = embed(graph.embedder, name)

    # Fetch all nodes matching filters that have embeddings
    (fclause, fparams) = _filter_clause(filters, "n")
    query = "MATCH (n) WHERE n.embedding IS NOT NULL$(fclause) RETURN n.name AS name, n.embedding AS embedding"
    result = _cypher_query(graph, query, fparams)

    best_name = nothing
    best_score = -Inf
    for row in result.rows
        node_name = row[1]
        node_emb = row[2]
        node_emb === nothing && continue
        score = cosine_similarity(target_embedding, Float64.(node_emb))
        if score >= th && score > best_score
            best_score = score
            best_name = node_name
        end
    end
    return best_name
end

function _get_or_create_node!(graph::Neo4jGraphStore, name::String,
                               entity_type::String, filters::Dict)::String
    norm_name = normalize_entity(name)

    existing = _find_similar_node(graph, norm_name, filters)
    existing !== nothing && return existing

    # MERGE node on name + filters
    node_embedding = embed(graph.embedder, norm_name)
    (fclause, fparams) = _filter_clause(filters, "n")
    set_filters = _filter_set_clause(filters, "n")
    fp = _filter_params(filters)

    query = """
    MERGE (n {name: \$name$(isempty(filters) ? "" : ", " * join(["$(k): \$f_$(k)" for (k,_) in filters], ", "))})
    ON CREATE SET n.entity_type = \$entity_type,
                  n.embedding = \$embedding,
                  n.mentions = 1,
                  n.created_at = datetime()$(set_filters)
    ON MATCH SET n.mentions = coalesce(n.mentions, 0) + 1
    RETURN n.name AS name
    """
    params = merge(Dict{String, Any}(
        "name" => norm_name,
        "entity_type" => entity_type,
        "embedding" => node_embedding,
    ), fp)

    _cypher_query(graph, query, params)
    return norm_name
end

# ── Core graph operations ────────────────────────────────────────────

function add_to_graph!(graph::Neo4jGraphStore, data::String, filters::Dict)
    entities = _extract_entities(graph, data)
    user_id = get(filters, "user_id", nothing)
    relations = _extract_relations(graph, data; user_id=user_id)

    for rel in relations
        src_type = get(entities, normalize_entity(rel["source"]), "unknown")
        dst_type = get(entities, normalize_entity(rel["destination"]), "unknown")

        src_name = _get_or_create_node!(graph, rel["source"], src_type, filters)
        dst_name = _get_or_create_node!(graph, rel["destination"], dst_type, filters)

        rel_type = uppercase(replace(rel["relationship"], " " => "_"))

        # MERGE relationship between the two nodes
        (fclause_s, fps) = _filter_clause(filters, "s", prefix="fs")
        (fclause_d, fpd) = _filter_clause(filters, "d", prefix="fd")

        query = """
        MATCH (s {name: \$src_name}) WHERE true$(fclause_s)
        MATCH (d {name: \$dst_name}) WHERE true$(fclause_d)
        MERGE (s)-[r:$(rel_type)]->(d)
        ON CREATE SET r.created_at = datetime(), r.updated_at = datetime(),
                      r.mentions = 1, r.valid = true
        ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1,
                     r.valid = true, r.updated_at = datetime()
        RETURN s.name AS source, type(r) AS relationship, d.name AS destination
        """
        params = merge(Dict{String, Any}(
            "src_name" => src_name,
            "dst_name" => dst_name,
        ), fps, fpd)

        _cypher_query(graph, query, params)
    end

    return relations
end

function search_graph(graph::Neo4jGraphStore, query::String, filters::Dict; limit::Int=100)
    query_embedding = embed(graph.embedder, normalize_entity(query))

    # Fetch all nodes matching filters with embeddings
    (fclause, fparams) = _filter_clause(filters, "n")
    cypher = "MATCH (n) WHERE n.embedding IS NOT NULL$(fclause) RETURN n.name AS name, n.embedding AS embedding"
    result = _cypher_query(graph, cypher, fparams)

    # Client-side cosine similarity
    scored = Tuple{String, Float64}[]
    for row in result.rows
        node_name = row[1]
        node_emb = row[2]
        node_emb === nothing && continue
        score = cosine_similarity(query_embedding, Float64.(node_emb))
        score >= graph.threshold && push!(scored, (node_name, score))
    end
    sort!(scored, by=x -> -x[2])
    matched = Set(first.(scored[1:min(limit, length(scored))]))
    isempty(matched) && return Dict{String, Any}[]

    # Fetch relationships connected to matched nodes
    (fclause_s, fps) = _filter_clause(filters, "s", prefix="fs")
    (fclause_d, fpd) = _filter_clause(filters, "d", prefix="fd")

    names_list = collect(matched)
    rel_query = """
    MATCH (s)-[r]->(d)
    WHERE (s.name IN \$names OR d.name IN \$names)
      AND (r.valid IS NULL OR r.valid = true)$(fclause_s)$(fclause_d)
    RETURN DISTINCT s.name AS source, type(r) AS relationship, d.name AS destination
    LIMIT \$lim
    """
    rel_params = merge(Dict{String, Any}("names" => names_list, "lim" => limit), fps, fpd)
    rel_result = _cypher_query(graph, rel_query, rel_params)

    results = Dict{String, Any}[]
    for row in rel_result.rows
        push!(results, Dict{String, Any}(
            "source" => row[1],
            "relationship" => row[2],
            "destination" => row[3],
        ))
    end
    return results
end

function delete_from_graph!(graph::Neo4jGraphStore, data::String, filters::Dict)
    (fclause_s, fps) = _filter_clause(filters, "s", prefix="fs")
    (fclause_d, fpd) = _filter_clause(filters, "d", prefix="fd")

    query = """
    MATCH (s)-[r]->(d)
    WHERE (r.valid IS NULL OR r.valid = true)$(fclause_s)$(fclause_d)
    SET r.valid = false, r.invalidated_at = datetime()
    RETURN count(r) AS deleted_count
    """
    _cypher_query(graph, query, merge(fps, fpd))
end

function delete_all_graph!(graph::Neo4jGraphStore, filters::Dict)
    if isempty(filters)
        # Safety: refuse to DETACH DELETE everything without filters
        throw(Mem0Error("delete_all_graph! requires at least one filter"))
    end
    (fclause, fparams) = _filter_clause(filters, "n")
    query = "MATCH (n) WHERE true$(fclause) DETACH DELETE n"
    _cypher_query(graph, query, fparams)
end

function get_all_graph(graph::Neo4jGraphStore, filters::Dict; limit::Int=100)
    (fclause_s, fps) = _filter_clause(filters, "s", prefix="fs")
    (fclause_d, fpd) = _filter_clause(filters, "d", prefix="fd")

    query = """
    MATCH (s)-[r]->(d)
    WHERE (r.valid IS NULL OR r.valid = true)$(fclause_s)$(fclause_d)
    RETURN s.name AS source, type(r) AS relationship, d.name AS destination
    LIMIT \$lim
    """
    result = _cypher_query(graph, query, merge(Dict{String,Any}("lim" => limit), fps, fpd))

    results = Dict{String, Any}[]
    for row in result.rows
        push!(results, Dict{String, Any}(
            "source" => row[1],
            "relationship" => row[2],
            "destination" => row[3],
        ))
    end
    return results
end
