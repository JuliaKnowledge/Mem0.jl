# Main Memory class — the core orchestration layer

"""
    Memory

The main Mem0 memory system. Orchestrates LLM-based fact extraction,
embedding, vector search, optional graph memory, and history tracking.
"""
mutable struct Memory
    config::MemoryConfig
    llm::AbstractLLM
    embedder::AbstractEmbedder
    vector_store::AbstractVectorStore
    db::HistoryManager
    graph::Union{Nothing, AbstractGraphStore}
    enable_graph::Bool
    collection_name::String
    custom_fact_extraction_prompt::Union{Nothing, String}
    custom_update_memory_prompt::Union{Nothing, String}
end

"""
    Memory(; config=MemoryConfig())

Create a new Memory instance with the given configuration.
"""
function Memory(; config::MemoryConfig=MemoryConfig())
    llm = create_llm(config.llm)
    embedder = create_embedder(config.embedder)
    vector_store = create_vector_store(config.vector_store)
    db = HistoryManager(config.history_db_path)
    collection_name = get(config.vector_store.config, "collection_name", "mem0")

    # Initialize graph store if configured
    graph = nothing
    enable_graph = false
    if config.graph_store.config !== nothing
        graph = create_graph_store(config.graph_store, config.llm, config.embedder)
        enable_graph = true
    end

    return Memory(config, llm, embedder, vector_store, db,
                  graph, enable_graph, collection_name,
                  config.custom_fact_extraction_prompt,
                  config.custom_update_memory_prompt)
end

# --- Public API ---

"""
    add(mem, messages; user_id=nothing, agent_id=nothing, run_id=nothing,
        metadata=nothing, infer=true, memory_type=nothing, prompt=nothing)

Create new memories from messages. When `infer=true` (default), the LLM extracts
facts from the conversation, deduplicates against existing memories, and decides
whether to ADD, UPDATE, or DELETE. When `infer=false`, messages are stored directly.

Returns a `Dict` with "results" and optionally "relations" keys.
"""
function add(mem::Memory, messages;
             user_id::Union{Nothing, String}=nothing,
             agent_id::Union{Nothing, String}=nothing,
             run_id::Union{Nothing, String}=nothing,
             metadata::Union{Nothing, Dict}=nothing,
             infer::Bool=true,
             memory_type::Union{Nothing, String}=nothing,
             prompt::Union{Nothing, String}=nothing)

    processed_metadata, effective_filters = build_filters_and_metadata(
        user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata)

    # Handle procedural memory
    if agent_id !== nothing && memory_type == "procedural_memory"
        results = _create_procedural_memory(mem, messages; metadata=processed_metadata, prompt=prompt)
        return Dict{String, Any}("results" => results)
    end

    # Parse messages to string
    msg_text = messages isa AbstractString ? messages : parse_messages(messages)

    # Add to vector store
    vector_results = _add_to_vector_store(mem, msg_text, processed_metadata, effective_filters, infer)

    # Add to graph if enabled
    graph_results = nothing
    if mem.enable_graph && mem.graph !== nothing
        try
            graph_results = add_to_graph!(mem.graph, msg_text, effective_filters)
        catch e
            @warn "Graph memory add failed" exception=(e, catch_backtrace())
        end
    end

    result = Dict{String, Any}("results" => vector_results)
    if mem.enable_graph
        result["relations"] = something(graph_results, Dict{String, Any}[])
    end
    return result
end

"""
    search(mem, query; user_id=nothing, agent_id=nothing, run_id=nothing,
           limit=100, filters=nothing, threshold=nothing)

Search memories by semantic similarity.

Returns a `Dict` with "results" and optionally "relations" keys.
"""
function search(mem::Memory, query::String;
                user_id::Union{Nothing, String}=nothing,
                agent_id::Union{Nothing, String}=nothing,
                run_id::Union{Nothing, String}=nothing,
                limit::Int=100,
                filters::Union{Nothing, Dict}=nothing,
                threshold::Union{Nothing, Float64}=nothing)

    _, effective_filters = build_filters_and_metadata(
        user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters)

    # Search vector store
    vector_results = _search_vector_store(mem, query, effective_filters, limit, threshold)

    # Search graph if enabled
    graph_results = nothing
    if mem.enable_graph && mem.graph !== nothing
        try
            graph_results = search_graph(mem.graph, query, effective_filters; limit=limit)
        catch e
            @warn "Graph memory search failed" exception=(e, catch_backtrace())
        end
    end

    result = Dict{String, Any}("results" => vector_results)
    if mem.enable_graph
        result["relations"] = something(graph_results, Dict{String, Any}[])
    end
    return result
end

"""
    update(mem, memory_id, data; metadata=nothing)

Update an existing memory's content and optionally its metadata.
"""
function update(mem::Memory, memory_id::String, data::String;
                metadata::Union{Nothing, Dict}=nothing)
    existing = Base.get(mem.vector_store, memory_id)
    if existing === nothing
        throw(Mem0ValidationError("Memory with id $memory_id not found"))
    end

    new_embedding = embed(mem.embedder, data; memory_action="update")
    _update_memory(mem, memory_id, data, new_embedding, metadata)
    return Dict{String, Any}("message" => "Memory updated successfully!")
end

"""
    delete(mem, memory_id)

Delete a memory by ID.
"""
function delete(mem::Memory, memory_id::String)
    existing = Base.get(mem.vector_store, memory_id)
    if existing === nothing
        throw(Mem0ValidationError("Memory with id $memory_id not found"))
    end

    # Clean up graph if enabled
    if mem.enable_graph && mem.graph !== nothing
        try
            memory_text = get(existing.payload, "data", "")
            if !isempty(memory_text)
                gfilters = Dict{String, Any}()
                for key in ("user_id", "agent_id", "run_id")
                    val = get(existing.payload, key, nothing)
                    val !== nothing && (gfilters[key] = val)
                end
                !isempty(gfilters) && delete_from_graph!(mem.graph, memory_text, gfilters)
            end
        catch e
            @warn "Graph cleanup on delete failed" exception=(e, catch_backtrace())
        end
    end

    _delete_memory(mem, memory_id, existing)
    return Dict{String, Any}("message" => "Memory deleted successfully!")
end

"""
    get_memory(mem, memory_id)

Retrieve a single memory by ID.
"""
function get_memory(mem::Memory, memory_id::String)
    record = Base.get(mem.vector_store, memory_id)
    record === nothing && return nothing

    result = Dict{String, Any}(
        "id" => record.id,
        "memory" => get(record.payload, "data", ""),
        "hash" => get(record.payload, "hash", ""),
        "metadata" => record.payload,
        "created_at" => get(record.payload, "created_at", nothing),
        "updated_at" => get(record.payload, "updated_at", nothing),
    )

    # Promote session ID fields
    for key in ("user_id", "agent_id", "run_id", "actor_id", "role")
        result[key] = get(record.payload, key, nothing)
    end

    return result
end

"""
    get_all(mem; user_id=nothing, agent_id=nothing, run_id=nothing,
            filters=nothing, limit=100)

Get all memories for given session IDs.
"""
function get_all(mem::Memory;
                 user_id::Union{Nothing, String}=nothing,
                 agent_id::Union{Nothing, String}=nothing,
                 run_id::Union{Nothing, String}=nothing,
                 filters::Union{Nothing, Dict}=nothing,
                 limit::Int=100)

    _, effective_filters = build_filters_and_metadata(
        user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters)

    records = list_records(mem.vector_store; filters=effective_filters, limit=limit)

    results = Dict{String, Any}[]
    for record in records
        entry = Dict{String, Any}(
            "id" => record.id,
            "memory" => get(record.payload, "data", ""),
            "hash" => get(record.payload, "hash", ""),
            "metadata" => record.payload,
            "created_at" => get(record.payload, "created_at", nothing),
            "updated_at" => get(record.payload, "updated_at", nothing),
        )
        for key in ("user_id", "agent_id", "run_id", "actor_id", "role")
            entry[key] = get(record.payload, key, nothing)
        end
        push!(results, entry)
    end

    result = Dict{String, Any}("results" => results)

    if mem.enable_graph && mem.graph !== nothing
        try
            graph_rels = get_all_graph(mem.graph, effective_filters; limit=limit)
            result["relations"] = graph_rels
        catch e
            @warn "Graph get_all failed" exception=(e, catch_backtrace())
            result["relations"] = Dict{String, Any}[]
        end
    end

    return result
end

"""
    history(mem, memory_id)

Get the change history for a specific memory.
"""
function history(mem::Memory, memory_id::String)
    return get_history(mem.db, memory_id)
end

"""
    reset!(mem)

Reset all memories (vector store + graph + history).
"""
function reset!(mem::Memory)
    Mem0.reset!(mem.vector_store)
    if mem.enable_graph && mem.graph !== nothing
        # Reset graph by clearing all edges/nodes
        lock(mem.graph.lock) do
            empty!(mem.graph.nodes)
            empty!(mem.graph.edges)
        end
    end
    reset_history!(mem.db)
end

# --- Internal methods ---

function _add_to_vector_store(mem::Memory, msg_text::String,
                               metadata::Dict{String, Any},
                               filters::Dict{String, Any},
                               infer::Bool)
    if !infer
        # Direct storage without LLM inference
        return _store_raw_messages(mem, msg_text, metadata)
    end

    # Step 1: Extract facts using LLM
    is_agent = haskey(metadata, "agent_id") && !haskey(metadata, "user_id")
    system_prompt, user_content = get_fact_retrieval_messages(msg_text; is_agent_memory=is_agent)

    if mem.custom_fact_extraction_prompt !== nothing
        system_prompt = mem.custom_fact_extraction_prompt
    end

    llm_messages = [
        Dict("role" => "system", "content" => system_prompt),
        Dict("role" => "user", "content" => user_content),
    ]

    response = generate_response(mem.llm, llm_messages; response_format="json")
    content = response.content
    content === nothing && return Dict{String, Any}[]

    # Parse extracted facts
    parsed = extract_json(content)
    parsed === nothing && return Dict{String, Any}[]

    raw_facts = get(parsed, "facts", [])
    facts = normalize_facts(raw_facts)
    isempty(facts) && return Dict{String, Any}[]

    # Step 2: For each fact, search existing memories and decide action
    results = Dict{String, Any}[]

    # Build existing memories string for the update prompt
    existing_memories_str = ""
    fact_memory_map = Dict{String, Vector{Tuple{String, String, Float64}}}()

    embedding_cache = Dict{String, Vector{Float64}}()

    for fact in facts
        fact_embedding = embed(mem.embedder, fact; memory_action="add")
        embedding_cache[fact] = fact_embedding

        matches = Mem0.search(mem.vector_store, fact_embedding, 5;
                              filters=filters, threshold=0.5)
        fact_memory_map[fact] = [(m[1], get(m[3], "data", ""), m[2]) for m in matches]
    end

    # Build existing memory context
    existing_ids = Dict{String, String}()
    memory_lines = String[]
    for (fact, matches) in fact_memory_map
        for (mid, mtext, mscore) in matches
            if !haskey(existing_ids, mid)
                existing_ids[mid] = mtext
                push!(memory_lines, "- ID: $(mid) | Memory: $(mtext)")
            end
        end
    end
    existing_memories_str = isempty(memory_lines) ? "No existing memories." : join(memory_lines, "\n")

    new_facts_str = join(["- $f" for f in facts], "\n")

    # Step 3: Ask LLM to decide ADD/UPDATE/DELETE/NONE
    update_messages = get_update_memory_messages(existing_memories_str, new_facts_str;
                                                  custom_prompt=mem.custom_update_memory_prompt)
    update_response = generate_response(mem.llm, update_messages; response_format="json")

    update_content = update_response.content
    update_content === nothing && return Dict{String, Any}[]

    update_parsed = extract_json(update_content)
    update_parsed === nothing && return Dict{String, Any}[]

    operations = get(update_parsed, "memory", [])

    for op in operations
        event = uppercase(get(op, "event", "NONE"))
        text = get(op, "text", "")
        op_id = get(op, "id", "")

        if event == "ADD" && !isempty(text)
            result = _create_memory(mem, text, metadata, embedding_cache)
            push!(results, result)
        elseif event == "UPDATE" && !isempty(text) && haskey(existing_ids, op_id)
            vec = get(embedding_cache, text, embed(mem.embedder, text; memory_action="update"))
            _update_memory(mem, op_id, text, vec, metadata)
            push!(results, Dict{String, Any}("id" => op_id, "event" => "UPDATE", "memory" => text))
        elseif event == "DELETE" && haskey(existing_ids, op_id)
            record = Base.get(mem.vector_store, op_id)
            if record !== nothing
                _delete_memory(mem, op_id, record)
                push!(results, Dict{String, Any}("id" => op_id, "event" => "DELETE"))
            end
        end
    end

    return results
end

function _store_raw_messages(mem::Memory, msg_text::String, metadata::Dict{String, Any})
    return [_create_memory(mem, msg_text, metadata, Dict{String, Vector{Float64}}())]
end

function _search_vector_store(mem::Memory, query::String, filters::Dict,
                               limit::Int, threshold::Union{Nothing, Float64})
    query_embedding = embed(mem.embedder, query; memory_action="search")
    matches = Mem0.search(mem.vector_store, query_embedding, limit;
                          filters=filters, threshold=threshold)

    results = Dict{String, Any}[]
    for (mid, score, payload) in matches
        entry = Dict{String, Any}(
            "id" => mid,
            "memory" => get(payload, "data", ""),
            "hash" => get(payload, "hash", ""),
            "score" => score,
            "metadata" => payload,
            "created_at" => get(payload, "created_at", nothing),
            "updated_at" => get(payload, "updated_at", nothing),
        )
        for key in ("user_id", "agent_id", "run_id", "actor_id", "role")
            entry[key] = get(payload, key, nothing)
        end
        push!(results, entry)
    end

    return results
end

function _create_memory(mem::Memory, text::String, metadata::Dict{String, Any},
                         embedding_cache::Dict{String, Vector{Float64}})
    memory_id = string(uuid4())
    hash = memory_hash(text)
    ts = now_iso()

    payload = copy(metadata)
    payload["data"] = text
    payload["hash"] = hash
    payload["created_at"] = ts
    payload["updated_at"] = ts

    vec = get(embedding_cache, text, embed(mem.embedder, text; memory_action="add"))

    Base.insert!(mem.vector_store, vec, payload, memory_id)

    add_history!(mem.db, memory_id, nothing, text, "ADD";
                 created_at=ts, updated_at=ts)

    return Dict{String, Any}("id" => memory_id, "event" => "ADD", "memory" => text)
end

function _update_memory(mem::Memory, memory_id::String, new_text::String,
                         new_embedding::AbstractVector,
                         new_metadata::Union{Nothing, Dict})
    existing = Base.get(mem.vector_store, memory_id)
    old_text = existing !== nothing ? get(existing.payload, "data", "") : ""
    ts = now_iso()

    payload_update = Dict{String, Any}(
        "data" => new_text,
        "hash" => memory_hash(new_text),
        "updated_at" => ts,
    )

    if new_metadata !== nothing
        merge!(payload_update, new_metadata)
    end

    Mem0.update!(mem.vector_store, memory_id;
                 vector=Float64.(new_embedding),
                 payload=payload_update)

    add_history!(mem.db, memory_id, old_text, new_text, "UPDATE";
                 created_at=ts, updated_at=ts)
end

function _delete_memory(mem::Memory, memory_id::String, existing::VectorRecord)
    old_text = get(existing.payload, "data", "")
    ts = now_iso()

    Base.delete!(mem.vector_store, memory_id)

    add_history!(mem.db, memory_id, old_text, nothing, "DELETE";
                 created_at=ts, updated_at=ts, is_deleted=1)
end

function _create_procedural_memory(mem::Memory, messages;
                                    metadata::Dict{String, Any}=Dict{String, Any}(),
                                    prompt::Union{Nothing, String}=nothing)
    msg_text = messages isa AbstractString ? messages : parse_messages(messages)

    sys_prompt = something(prompt, PROCEDURAL_MEMORY_SYSTEM_PROMPT)
    llm_messages = [
        Dict("role" => "system", "content" => sys_prompt),
        Dict("role" => "user", "content" => "Record the following conversation:\n\n$msg_text"),
    ]

    response = generate_response(mem.llm, llm_messages)
    summary = something(response.content, msg_text)

    return [_create_memory(mem, summary, metadata, Dict{String, Vector{Float64}}())]
end
