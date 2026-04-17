module Mem0AzureIdentityExt

using Mem0
using AzureIdentity

# Resolve Mem0 OpenAI / Azure-OpenAI bearer tokens from an AzureIdentity
# credential. Scope defaults to the Cognitive Services scope used by
# Azure OpenAI, but callers can pass a different one.
function Mem0._resolve_bearer(cred::AzureIdentity.AbstractAzureCredential,
                              scope::AbstractString = Mem0.AZURE_COGNITIVE_SERVICES_SCOPE)
    return AzureIdentity.get_token(cred, String(scope)).token
end

end # module Mem0AzureIdentityExt
