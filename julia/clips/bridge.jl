# Bridge between CLIPS.jl (external/CLIPS.jl, never edited) and the table of local language-model scores.
#
# Include this file in Main before any CLIPS src file. It defines `GPT3Mixture` in Main, so when CLIPS's
# src/utterances.jl runs `using GenGPT3` and then `GPT3Mixture(model=..., ...)`, Julia keeps Main's
# binding (it warns that GenGPT3's export conflicts) and each of the four mixtures CLIPS defines is a
# TableMixture with its role:
#
#   domain  CLIPS mixture            model              temperature  role
#   mdkg    pragmatic_gpt3_mixture   curie              1.0          mdkg_pragmatic
#   mdkg    literal_gpt3_mixture     text-davinci-002   1.0          mdkg_literal
#   vh      pragmatic_gpt3_mixture   davinci-002        1.0          vh_pragmatic
#   vh      literal_gpt3_mixture     davinci-002        0.5          vh_literal
#
# This has the effect of rebinding the two globals after CLIPS's src files are included (plan 2.2), with
# the src files loaded unchanged. The mode comes from the environment:
#   CLIPS_MODE        record (default) or replay
#   CLIPS_TABLE       replay: the score table (JSON lines)
#   CLIPS_RECORD_LOG  record: where every asked (role, prompt, output) is appended
#   CLIPS_RESULTS     where the experiment loop writes CLIPS's own result files
#   CLIPS_PLANS       optional comma-separated plan ids (smoke runs)

include(joinpath(@__DIR__, "table_mixture.jl"))

const CLIPS_ROOT = normpath(joinpath(@__DIR__, "..", "..", "external", "CLIPS.jl"))
const CLIPS_DOMAIN = Ref("")
const MIXTURE_ROLES = Dict(
    ("mdkg", "curie", 1.0) => "mdkg_pragmatic",
    ("mdkg", "text-davinci-002", 1.0) => "mdkg_literal",
    ("vh", "davinci-002", 1.0) => "vh_pragmatic",
    ("vh", "davinci-002", 0.5) => "vh_literal",
)

"Stands in for GenGPT3.GPT3Mixture: the same keyword call returns the TableMixture of its role."
function GPT3Mixture(; model::AbstractString, temperature::Real=1.0, kwargs...)
    key = (CLIPS_DOMAIN[], String(model), Float64(temperature))
    haskey(MIXTURE_ROLES, key) || error("no TableMixture role for $key")
    return TableMixture(MIXTURE_ROLES[key]; temperature=temperature)
end

function configure_from_env!()
    mode = Symbol(get(ENV, "CLIPS_MODE", "record"))
    configure!(mode; table_path=get(ENV, "CLIPS_TABLE", nothing), record_path=get(ENV, "CLIPS_RECORD_LOG", nothing))
    return mode
end

"The plan ids to run: all of them, or those named in CLIPS_PLANS."
function chosen_plans(plan_ids)
    wanted = strip(get(ENV, "CLIPS_PLANS", ""))
    isempty(wanted) && return plan_ids
    names = Set(strip.(split(wanted, ",")))
    found = filter(id -> id in names, plan_ids)
    length(found) == length(names) || error("CLIPS_PLANS names unknown plans: $(setdiff(names, Set(found)))")
    return found
end

results_dir(domain::AbstractString) =
    get(ENV, "CLIPS_RESULTS", normpath(joinpath(@__DIR__, "..", "..", "results", "clips", "unnamed", domain)))

"""Record mode only: each particle's goal, cost profile and its utterance model's prompts (with their
weights) at every utterance time, to \$CLIPS_PARTICLE_LOG. The prompts are those enumerate_commands made
from that particle's plan, so this is the per-goal command record of plan Phase 1."""
function record_particles(pf, utterance_times)
    path = get(ENV, "CLIPS_PARTICLE_LOG", nothing)
    isnothing(path) && return nothing
    open(path, "a") do io
        for (index, trace) in enumerate(pf.traces)
            goal = trace[:init => :agent => :goal => :goal]
            cost = trace[:init => :agent => :goal => :cost_idx]
            for t in utterance_times
                step = t == 0 ? Gen.static_get_subtrace(trace, Val(:init)) :
                    Gen.static_get_subtrace(trace, Val(:timestep)).subtraces[t]
                act_trace = step.trie[:act].subtrace_or_retval
                utt = act_trace.trie[:utterance].subtrace_or_retval
                JSON3.write(io, Dict("plan_id" => get(CONTEXT, "plan_id", ""), "particle" => index, "goal" => goal,
                                     "cost" => cost, "t" => t, "prompts" => utt.prompts, "probs" => utt.probs,
                                     "output" => utt.output))
                write(io, "\n")
            end
        end
    end
    return nothing
end
