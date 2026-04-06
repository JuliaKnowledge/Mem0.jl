# SQLite-based history tracking for memory operations

using SQLite
using SQLite: DBInterface
using UUIDs

"""
    HistoryManager

Thread-safe SQLite-based history tracker for memory operations (add, update, delete).
"""
mutable struct HistoryManager
    db::SQLite.DB
    lock::ReentrantLock

    function HistoryManager(db_path::String=":memory:")
        db = SQLite.DB(db_path)
        mgr = new(db, ReentrantLock())
        _create_history_table!(mgr)
        return mgr
    end
end

function _create_history_table!(mgr::HistoryManager)
    lock(mgr.lock) do
        SQLite.execute(mgr.db, """
            CREATE TABLE IF NOT EXISTS history (
                id           TEXT PRIMARY KEY,
                memory_id    TEXT,
                old_memory   TEXT,
                new_memory   TEXT,
                event        TEXT,
                created_at   TEXT,
                updated_at   TEXT,
                is_deleted   INTEGER DEFAULT 0,
                actor_id     TEXT,
                role         TEXT
            )
        """)
    end
end

"""
    add_history!(mgr, memory_id, old_memory, new_memory, event; kwargs...)

Insert a history record for a memory operation.
"""
function add_history!(mgr::HistoryManager, memory_id::String,
                      old_memory::Union{Nothing, String},
                      new_memory::Union{Nothing, String},
                      event::String;
                      created_at::Union{Nothing, String}=nothing,
                      updated_at::Union{Nothing, String}=nothing,
                      is_deleted::Int=0,
                      actor_id::Union{Nothing, String}=nothing,
                      role::Union{Nothing, String}=nothing)
    lock(mgr.lock) do
        SQLite.execute(mgr.db, """
            INSERT INTO history (id, memory_id, old_memory, new_memory, event,
                                 created_at, updated_at, is_deleted, actor_id, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            string(uuid4()),
            memory_id,
            old_memory === nothing ? missing : old_memory,
            new_memory === nothing ? missing : new_memory,
            event,
            created_at === nothing ? missing : created_at,
            updated_at === nothing ? missing : updated_at,
            is_deleted,
            actor_id === nothing ? missing : actor_id,
            role === nothing ? missing : role,
        ])
    end
end

"""
    get_history(mgr, memory_id)

Retrieve all history records for a given memory ID, ordered chronologically.
"""
function get_history(mgr::HistoryManager, memory_id::String)::Vector{Dict{String, Any}}
    results = Dict{String, Any}[]
    lock(mgr.lock) do
        stmt = SQLite.Stmt(mgr.db, """
            SELECT id, memory_id, old_memory, new_memory, event,
                   created_at, updated_at, is_deleted, actor_id, role
            FROM history
            WHERE memory_id = ?
            ORDER BY created_at ASC, updated_at ASC
        """)
        q = DBInterface.execute(stmt, [memory_id])
        for row in q
            push!(results, Dict{String, Any}(
                "id" => row[1],
                "memory_id" => row[2],
                "old_memory" => row[3],
                "new_memory" => row[4],
                "event" => row[5],
                "created_at" => row[6],
                "updated_at" => row[7],
                "is_deleted" => row[8] == 1,
                "actor_id" => row[9],
                "role" => row[10],
            ))
        end
    end
    return results
end

"""
    reset_history!(mgr)

Drop and recreate the history table.
"""
function reset_history!(mgr::HistoryManager)
    lock(mgr.lock) do
        SQLite.execute(mgr.db, "DROP TABLE IF EXISTS history")
    end
    _create_history_table!(mgr)
end

"""
    close_history!(mgr)

Close the SQLite database connection.
"""
function close_history!(mgr::HistoryManager)
    # SQLite.jl handles finalization automatically
end
