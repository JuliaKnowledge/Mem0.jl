# Abstract graph store interface

"""
    AbstractGraphStore

Abstract base type for graph memory backends.
All concrete subtypes must have `llm::AbstractLLM` and
`custom_prompt::Union{Nothing, String}` fields.
"""
abstract type AbstractGraphStore end

"""
    add_to_graph!(graph, data, filters)

Add entities and relationships extracted from data into the graph.
"""
function add_to_graph! end

"""
    search_graph(graph, query, filters; limit=100)

Search the graph for entities and relationships relevant to the query.
"""
function search_graph end

"""
    delete_from_graph!(graph, data, filters)

Soft-delete relationships matching the data.
"""
function delete_from_graph! end

"""
    delete_all_graph!(graph, filters)

Delete all entities/relationships matching the filters.
"""
function delete_all_graph! end

"""
    get_all_graph(graph, filters; limit=100)

Retrieve all relationships matching filters.
"""
function get_all_graph end

# --- Shared LLM-based entity / relation extraction ---

"""
    _extract_entities(graph::AbstractGraphStore, text) → Dict{String, String}

Use the graph store's LLM to extract named entities from text.
Returns a `name => entity_type` mapping.
"""
function _extract_entities(graph::AbstractGraphStore, text::String)
    messages = [
        Dict("role" => "system", "content" => "Extract entities and their types from the given text."),
        Dict("role" => "user", "content" => text),
    ]
    resp = generate_response(graph.llm, messages; tools=EXTRACT_ENTITIES_TOOL, tool_choice="auto")

    entities = Dict{String, String}()
    for tc in resp.tool_calls
        ents = get(tc["arguments"], "entities", [])
        for e in ents
            name = normalize_entity(get(e, "entity", ""))
            etype = get(e, "entity_type", "unknown")
            !isempty(name) && (entities[name] = etype)
        end
    end

    # Fallback: try to parse from content if no tool calls
    if isempty(entities) && resp.content !== nothing
        parsed = extract_json(resp.content)
        if parsed !== nothing
            for e in get(parsed, "entities", [])
                name = normalize_entity(get(e, "entity", ""))
                etype = get(e, "entity_type", "unknown")
                !isempty(name) && (entities[name] = etype)
            end
        end
    end

    return entities
end

"""
    _extract_relations(graph::AbstractGraphStore, text; user_id=nothing) → Vector{Dict}

Use the graph store's LLM to extract relationships from text.
Each result is `Dict("source" => …, "relationship" => …, "destination" => …)`.
"""
function _extract_relations(graph::AbstractGraphStore, text::String;
                             user_id::Union{Nothing, String}=nothing)
    sys_prompt = extract_relations_prompt(user_id=user_id, custom_prompt=graph.custom_prompt)
    messages = [
        Dict("role" => "system", "content" => sys_prompt),
        Dict("role" => "user", "content" => text),
    ]
    resp = generate_response(graph.llm, messages; tools=EXTRACT_RELATIONS_TOOL, tool_choice="auto")

    relations = Dict{String, Any}[]
    for tc in resp.tool_calls
        rels = get(tc["arguments"], "relationships", [])
        for r in rels
            src = normalize_entity(get(r, "source", ""))
            rel = replace(strip(get(r, "relationship", "")), " " => "_")
            dst = normalize_entity(get(r, "destination", ""))
            if !isempty(src) && !isempty(rel) && !isempty(dst)
                push!(relations, Dict{String, Any}("source" => src, "relationship" => rel, "destination" => dst))
            end
        end
    end

    # Fallback: parse from LLM content
    if isempty(relations) && resp.content !== nothing
        parsed = extract_json(resp.content)
        if parsed !== nothing
            for r in get(parsed, "relationships", get(parsed, "relations", []))
                src = normalize_entity(get(r, "source", ""))
                rel = replace(strip(get(r, "relationship", "")), " " => "_")
                dst = normalize_entity(get(r, "destination", ""))
                if !isempty(src) && !isempty(rel) && !isempty(dst)
                    push!(relations, Dict{String, Any}("source" => src, "relationship" => rel, "destination" => dst))
                end
            end
        end
    end

    return relations
end
