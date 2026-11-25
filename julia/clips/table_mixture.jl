# TableMixture: the mixture over prompts that CLIPS uses as its utterance likelihood, with the language
# model's scores read from a table instead of the OpenAI API (docs/clips_small_lm_plan.md 2.2).
#
# Given prompts p_1 ... p_n with weights w_i and the observed output u, the score is
#     logsumexp_i( table[(role, p_i, u)] / temperature + log w_i ).
# The trace has the fields and getindex addresses of GenGPT3.GPT3MixtureTrace that CLIPS reads
# (`prompts`, `probs`, `output`, `scores`, `score`; :output, :post_probs, ...).
#
# Modes, set by `configure!`:
#   record  every (role, prompt, output) asked about is appended to the record log (once per process),
#           with the context the experiment loop set in `CONTEXT`; every score is 0.
#   replay  scores come from the table; a missing key is an error that names it, never a default.
#
# The table file is JSON lines: {"role": ..., "prompt": ..., "output": ..., "score": ...} with the score in
# nats (log P(output + "\n" | prompt)), or null for -Inf (a continuation longer than max_tokens).

using Gen
using JSON3

const OUTPUT_ADDR = :output

mutable struct TableStore
    mode::Symbol
    table::Dict{Tuple{String,String,String},Float64}
    record_path::Union{String,Nothing}
    recorded::Set{Tuple{String,String,String}}
end

const STORE = TableStore(:record, Dict{Tuple{String,String,String},Float64}(), nothing,
                         Set{Tuple{String,String,String}}())
const CONTEXT = Dict{String,Any}()

"Set the mode; replay reads `table_path`, record appends to `record_path`."
function configure!(mode::Symbol; table_path=nothing, record_path=nothing)
    mode in (:record, :replay) || error("unknown TableMixture mode $mode")
    STORE.mode = mode
    empty!(STORE.table)
    empty!(STORE.recorded)
    STORE.record_path = record_path
    if mode == :replay
        isnothing(table_path) && error("replay needs a table")
        for line in eachline(table_path)
            isempty(strip(line)) && continue
            row = JSON3.read(line)
            score = isnothing(row.score) ? -Inf : Float64(row.score)
            STORE.table[(String(row.role), String(row.prompt), String(row.output))] = score
        end
    elseif !isnothing(record_path)
        mkpath(dirname(record_path))
        touch(record_path)
    end
    return STORE
end

function lookup(role::String, prompt::String, output::String)
    key = (role, prompt, output)
    if STORE.mode == :record
        if !(key in STORE.recorded)
            push!(STORE.recorded, key)
            if !isnothing(STORE.record_path)
                open(STORE.record_path, "a") do io
                    JSON3.write(io, Dict("role" => role, "prompt" => prompt, "output" => output,
                                         "context" => Dict(string(k) => string(v) for (k, v) in CONTEXT)))
                    write(io, "\n")
                end
            end
        end
        return 0.0
    end
    haskey(STORE.table, key) || error("TableMixture: no score for role $(repr(role)), output $(repr(output)), " *
                                      "prompt ending $(repr(last(prompt, 120)))")
    return STORE.table[key]
end

## Trace ##

struct TableMixtureTrace{T <: GenerativeFunction} <: Trace
    gen_fn::T
    prompts::Vector{String}
    probs::Vector{Float64}
    output::String
    tokens::Vector{String}
    scores::Vector{Float64}
    score::Float64
end

Gen.get_choices(trace::TableMixtureTrace) = Gen.choicemap((OUTPUT_ADDR, trace.output))
Gen.get_args(trace::TableMixtureTrace) = (trace.prompts, trace.probs)
Gen.get_retval(trace::TableMixtureTrace) = trace.output
Gen.get_score(trace::TableMixtureTrace) = trace.score
Gen.get_gen_fn(trace::TableMixtureTrace) = trace.gen_fn

function Base.getindex(trace::TableMixtureTrace, addr)
    if addr == :prompts
        return trace.prompts
    elseif addr == :prior_probs
        return trace.probs
    elseif addr == :post_probs
        return exp.(trace.scores .+ log.(trace.probs) .- trace.score)
    elseif addr == :joint_probs
        return exp.(trace.scores .+ log.(trace.probs))
    elseif addr == :tokens
        return trace.tokens
    elseif addr == :output
        return trace.output
    elseif addr == :output_scores
        return trace.scores
    elseif addr == :score
        return trace.score
    else
        throw(KeyError(addr))
    end
end

## Generative function ##

struct TableMixture <: GenerativeFunction{String,TableMixtureTrace}
    role::String
    temperature::Float64
end

TableMixture(role::AbstractString; temperature::Real=1.0) = TableMixture(String(role), Float64(temperature))

function mixture_args(args::Tuple)
    prompts = Vector{String}(args[1])
    probs = length(args) >= 2 ? Vector{Float64}(args[2]) : fill(1.0 / length(prompts), length(prompts))
    @assert length(prompts) == length(probs)
    return prompts, probs
end

function logsumexp_(values::Vector{Float64})
    top = maximum(values)
    isinf(top) && return top
    return top + log(sum(exp.(values .- top)))
end

function scored_trace(gen_fn::TableMixture, prompts, probs, output::String)
    scores = [lookup(gen_fn.role, prompt, output) / gen_fn.temperature for prompt in prompts]
    score = logsumexp_(scores .+ log.(probs))
    return TableMixtureTrace(gen_fn, prompts, probs, output, String[], scores, score)
end

function Gen.simulate(gen_fn::TableMixture, args::Tuple)
    error("TableMixture($(gen_fn.role)) cannot sample an utterance: there is no language model to sample from. " *
          "CLIPS only scores observed utterances; constrain :output.")
end

function Gen.generate(gen_fn::TableMixture, args::Tuple, constraints::ChoiceMap)
    if !has_value(constraints, OUTPUT_ADDR)
        return Gen.simulate(gen_fn, args), 0.0
    end
    prompts, probs = mixture_args(args)
    trace = scored_trace(gen_fn, prompts, probs, String(get_value(constraints, OUTPUT_ADDR)))
    return trace, trace.score
end

Gen.project(trace::TableMixtureTrace, selection::Selection) = OUTPUT_ADDR in selection ? trace.score : 0.0
Gen.project(trace::TableMixtureTrace, ::EmptySelection) = 0.0

function Gen.update(trace::TableMixtureTrace, args::Tuple, argdiffs::Tuple, constraints::ChoiceMap)
    if !isempty(constraints) && !has_value(constraints, OUTPUT_ADDR)
        error("Did not visit all constraints")
    end
    output = isempty(constraints) ? trace.output : String(get_value(constraints, OUTPUT_ADDR))
    prompts, probs = mixture_args(args)
    if isempty(constraints) && prompts == trace.prompts && probs == trace.probs
        return trace, 0.0, NoChange(), EmptyChoiceMap()
    end
    new_trace = scored_trace(trace.gen_fn, prompts, probs, output)
    retdiff = trace.output == output ? NoChange() : UnknownChange()
    discard = isempty(constraints) ? EmptyChoiceMap() : get_choices(trace)
    return new_trace, new_trace.score - trace.score, retdiff, discard
end

function Gen.regenerate(trace::TableMixtureTrace, args::Tuple, argdiffs::Tuple, selection::Selection)
    if !isempty(selection) && OUTPUT_ADDR in selection
        error("TableMixture($(trace.gen_fn.role)) cannot resample an utterance")
    end
    prompts, probs = mixture_args(args)
    if prompts == trace.prompts && probs == trace.probs
        return trace, 0.0, NoChange()
    end
    new_trace = scored_trace(trace.gen_fn, prompts, probs, trace.output)
    return new_trace, new_trace.score - trace.score, NoChange()
end

Gen.has_argument_grads(::TableMixture) = (false, false)
Gen.accepts_output_grad(::TableMixture) = false
