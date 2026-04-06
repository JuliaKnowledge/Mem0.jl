# Graph tools: LLM tool definitions and prompts for entity/relation extraction

const EXTRACT_ENTITIES_TOOL = [Dict{String, Any}(
    "type" => "function",
    "function" => Dict{String, Any}(
        "name" => "extract_entities",
        "description" => "Extract entities and their types from the given text.",
        "parameters" => Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "entities" => Dict{String, Any}(
                    "type" => "array",
                    "items" => Dict{String, Any}(
                        "type" => "object",
                        "properties" => Dict{String, Any}(
                            "entity" => Dict("type" => "string", "description" => "Name of the entity"),
                            "entity_type" => Dict("type" => "string", "description" => "Type/category of the entity"),
                        ),
                        "required" => ["entity", "entity_type"],
                    ),
                    "description" => "List of extracted entities",
                ),
            ),
            "required" => ["entities"],
        ),
    ),
)]

const EXTRACT_RELATIONS_TOOL = [Dict{String, Any}(
    "type" => "function",
    "function" => Dict{String, Any}(
        "name" => "establish_relationships",
        "description" => "Establish relationships between entities extracted from text.",
        "parameters" => Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "relationships" => Dict{String, Any}(
                    "type" => "array",
                    "items" => Dict{String, Any}(
                        "type" => "object",
                        "properties" => Dict{String, Any}(
                            "source" => Dict("type" => "string", "description" => "Source entity"),
                            "relationship" => Dict("type" => "string", "description" => "Relationship type"),
                            "destination" => Dict("type" => "string", "description" => "Destination entity"),
                        ),
                        "required" => ["source", "relationship", "destination"],
                    ),
                    "description" => "List of relationships",
                ),
            ),
            "required" => ["relationships"],
        ),
    ),
)]

const GRAPH_UPDATE_TOOL = [Dict{String, Any}(
    "type" => "function",
    "function" => Dict{String, Any}(
        "name" => "update_graph_memory",
        "description" => "Decide what graph memory operations to perform.",
        "parameters" => Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "operations" => Dict{String, Any}(
                    "type" => "array",
                    "items" => Dict{String, Any}(
                        "type" => "object",
                        "properties" => Dict{String, Any}(
                            "action" => Dict("type" => "string", "enum" => ["ADD", "UPDATE", "DELETE", "NOOP"]),
                            "source" => Dict("type" => "string"),
                            "destination" => Dict("type" => "string"),
                            "relationship" => Dict("type" => "string"),
                            "source_type" => Dict("type" => "string"),
                            "destination_type" => Dict("type" => "string"),
                        ),
                        "required" => ["action", "source", "destination", "relationship"],
                    ),
                ),
            ),
            "required" => ["operations"],
        ),
    ),
)]

function extract_relations_prompt(; user_id::Union{Nothing, String}=nothing,
                                     custom_prompt::Union{Nothing, String}=nothing)
    prompt = """You are a smart assistant tasked with extracting entities and their relationships from text.
Extract relationships in the form: source -- relationship_type -- destination.

Guidelines:
- Extract only explicitly stated relationships from the text.
- Use consistent, concise, and timeless relationship types (e.g., "works_at", "likes", "lives_in").
- Normalize entity names to lowercase with underscores replacing spaces.
- Do not infer relationships that are not directly stated."""

    if user_id !== nothing
        prompt *= "\n- When the text mentions \"I\" or \"me\", treat it as referring to the user \"$(user_id)\"."
    end

    if custom_prompt !== nothing
        prompt *= "\n\nAdditional instructions:\n$(custom_prompt)"
    end

    return prompt
end

const DELETE_RELATIONS_PROMPT = """You are a memory management system. Given existing graph relationships and new information,
determine which existing relationships should be DELETED.

Deletion criteria:
1. OUTDATED: The relationship is contradicted by newer information.
2. CONTRADICTORY: The new information directly conflicts with the existing relationship.

IMPORTANT: Do NOT delete a relationship if the same relationship type exists for different destinations.
For example, if "alice -- likes_food -- pizza" exists and new info says "alice likes burgers",
do NOT delete the pizza relationship. Only delete if the new info says "alice no longer likes pizza".

Return the relationships to delete as a list."""
