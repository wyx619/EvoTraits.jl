# Discrete models

[中文](discrete_zh.md) · [Index](README.md) · [SIMMAP](simmap.md)

`fit_mk` fits Mk models. `fit_corhmm` supports observed-state and hidden-rate models, ambiguous observations, custom rate structures and ancestral-state reconstruction.

```julia
using EvoTraits, Random
tree = load_tree("tree.nwk")
states = Dict("A" => "herb", "B" => "herb&tree",
              "C" => "tree", "D" => "?")
fit = fit_corhmm(tree, states; model = :ARD,
    state_order = ["herb", "tree"], rate_cat = 1,
    node_states = :marginal, Ntrials = 4, Nthreads = 4,
    max_iterations = 2000, rng = MersenneTwister(42))
fit.success
fit.transition_matrix
fit.asr
```

This example requires a tree containing the four named tips. `herb&tree` indicates an ambiguous/polymorphic observation, not a new Markov state; `?` allows all states. Explicit `state_order` fixes the meaning of Q rows and ASR columns. `node_states` may be `:marginal`, `:joint`, `:scaled` or `:none`; `asr_corhmm(fit; mode=:marginal)` performs reconstruction after fitting. `root_prior=:yang` is the default; root prior and root marginal posterior are different quantities.

`Ntrials` and `Nthreads` control fitting trials and requested parallel workers. Starting Julia with sufficient threads is necessary. Check fit diagnostics and convergence before [sampling histories](simmap.md).

## Model choices and outputs

`model` selects rate constraints such as `:ER`, `:SYM`, `:ARD`, `:SUEDE` or `:SRD`; the biological meaning of an ordered model depends on `state_order`. `rate_cat` is the number of hidden-rate classes. Use `rate_mat` to specify a custom index structure when the standard constraints do not represent the hypothesis.

`fit.transition_matrix` is the fitted Q matrix; `fit.index_matrix` identifies the parameter sharing pattern, and `fit.rates` stores fitted rates. `fit.root_prior_probs` is the root prior used for likelihood calculations, whereas `fit.asr` contains the requested reconstruction. `fit.diagnostics` includes missing and polymorphic tip counts, fixed-node information and other fit settings. Inspect `fit.fit.converged` as well as `fit.success`.

`tip_fog`, `fixed_nodes` and `lewis_asc_bias` are specialized options; use the Julia help mode and inspect the input assumptions before enabling them. For multi-character data, state combinations and forbidden or permitted simultaneous changes must be specified consistently with the biological question.

## Rate models and state order

For `k` observed states and one rate class:

| Model | Free rates | Constraint |
| --- | --- | --- |
| `:ER` | 1 | All permitted changes share a rate |
| `:SYM` | k(k−1)/2 | Forward/reverse rates are equal for each pair |
| `:ARD` | k(k−1) | Every directed change has its own rate |
| `:SUEDE` | 2 for an ordered chain | One shared forward and one shared reverse rate |
| `:SRD` | 2(k−1) | Each directed adjacent change has its own rate |

The ordered chain follows `state_order`. For example, `["herb", "shrub", "tree"]` under SRD forbids direct herb↔tree changes. Reversing the order preserves this adjacency graph; it changes parameter labels, rather than imposing a one-way evolutionary trend. Do not interpret a numeric state code as an ordinal biological variable unless that is the intended hypothesis.

## Ambiguity, missing taxa and hidden rates

A tip coded `"herb&tree"` has observation likelihood 1 in both states and 0 in other states. It is not a fourth state, nor does it prescribe a 50:50 posterior probability. A missing tip allows all states. A taxon absent from the state mapping is treated as missing.

With `rate_cat > 1`, the latent state space combines observed states and hidden-rate classes. Hidden classes can switch while the observed state remains unchanged. The fitted matrix and hidden ASR refer to this expanded space. Use `hidden_to_observed` and the observed ASR outputs to aggregate the hidden dimension; do not label hidden classes as observed biological categories.

## Important fitting options

| Keyword | Default | Meaning |
| --- | --- | --- |
| `model` | `:ARD` | Rate-sharing model |
| `rate_cat` | `1` | Hidden-rate class count |
| `state_order` | `nothing` | Explicit observed label order |
| `rate_mat` | `nothing` | Custom rate **index** matrix |
| `root_prior` | `:yang` | Root weighting for likelihood |
| `node_states` | `:marginal` | ASR method, or `:none` |
| `Ntrials`, `Nthreads` | `1`, `1` | Fitting trials and requested workers |
| `max_iterations` | `1_000_000` | Optimizer budget |
| `lower_bound`, `upper_bound` | `1e-9`, `100.0` | Rate bounds |
| `rel_tol` | `sqrt(eps(Float64))` | Relative stopping tolerance |
| `rng` | Default RNG | Initialization reproducibility |

Starting Julia with eight threads does not change the default `Ntrials=1`. For eight parallel trials, explicitly request `Ntrials=8, Nthreads=8`. More iterations do not guarantee a global optimum; compare restart likelihoods and inspect bounds.

Root weights can be `:yang`, `:flat`, `:maddfitz` or an explicit vector. `:yang` derives weights from Q; `:flat` supplies equal weights; `:maddfitz` uses likelihood-based weighting. None is itself the reconstructed probability of the root.

## Extracting the root posterior and Q

```julia
Q = fit.transition_matrix
# Q[i,j] is the rate from state i to state j; row sums are zero.
fit.index_matrix
fit.rates
fit.root_prior_probs

asr = asr_corhmm(fit; mode = :marginal)
i = findfirst(==(Int(fit.tree.root)), asr.node_ids)
root_posterior = asr.observed_likelihoods[i, :]
collect(zip(fit.observed_labels, root_posterior))
```

Locate the root by its ID, not by assuming that row one is the root. Marginal values integrate over states at other nodes. Joint reconstruction returns a best joint state configuration; scaled likelihoods must not be relabeled as marginal posteriors.

## Custom hypotheses and multiple characters

`rate_mat` contains parameter indices, not numerical Q values. Positive integer entries identify free rates; repeated indices share one parameter. Zero, negative, missing or non-finite off-diagonal entries prohibit changes. The diagonal is ignored. Prefer contiguous indices starting at 1.

A two-state asymmetric model has the structure:

```julia
index = [0 1; 2 0]
custom = fit_corhmm(tree, states;
    state_order = ["herb", "tree"], rate_mat = index,
    Ntrials = 4, Nthreads = 4, max_iterations = 2000)
```

For multiple character columns, the model represents combinations of character states. The default construction permits only one character to change at a time. An independent model ties appropriate rates across the other character's background; a dependent model releases these equalities. Simultaneous changes in both characters require an explicit custom structure. There is no general `dual=true` fitting keyword.

`tip_fog` adds observation uncertainty, `fixed_nodes` constrains specified ancestral states, and `lewis_asc_bias` conditions on variable-character sampling. These change the statistical model; do not enable them merely to improve likelihood. Their detailed input contracts are available through `?fit_corhmm`.

## Mk is a separate interface

`fit_mk` uses `rate_model`, not the corHMM keyword `model`. Its explicit-state-count interface accepts `tip_states` or `tip_priors`; the latter is useful for observation likelihoods. Do not copy all corHMM keywords into Mk calls. See the [runnable discrete example](examples/02_discrete_simmap.jl).

## Observation uncertainty and constrained nodes

| Option | Accepted form | Meaning |
| --- | --- | --- |
| `tip_fog=nothing` | Default | No added observation-error model |
| `tip_fog=0.01` | Scalar in [0,0.5) | Fixed common fog value |
| `tip_fog=[0.01,0.02]` | One probability per observed state | Fixed state-specific fog |
| `tip_fog=:estimate` | Symbol | Estimate one shared fog parameter |
| `tip_fog=[1,2]` | Positive integer group IDs | Estimate grouped fog parameters |
| `fixed_nodes=false` | Default | Do not constrain internal states |
| `fixed_nodes=Dict(node_id => "herb")` | Internal ID or node-label keys | Constrain specified observed ancestral states |
| `fixed_nodes=true` | Boolean | Read recognized state labels from internal node labels |
| `lewis_asc_bias=true` | Boolean | Condition likelihood on a character being variable |

Probability vectors and integer group vectors have different meanings; preserve their element types. Fog values modify observation likelihoods rather than add new Markov states. Entirely missing observations remain uninformative. For more than two states, valid individual fog entries must also leave positive mass for the allowed observed states.

Fixed-node keys are tree-specific, and tips are not accepted as fixed internal nodes. Use constraints only for an externally justified hypothesis. A root constraint deliberately fixes the answer, so it is unsuitable for a free test of the root state.

## Two binary characters: explicit index structures

For the declared combined-state order `["00", "10", "01", "11"]`, these structures encode independent, dependent single-change and dependent-with-dual hypotheses:

```julia
independent = [0 1 3 0;
               2 0 0 3;
               4 0 0 1;
               0 4 2 0]
dependent = [0 1 2 0;
             3 0 0 4;
             5 0 0 6;
             0 7 8 0]
dual = [0 1 2 9;
        3 0 10 4;
        5 11 0 6;
        12 7 8 0]
```

They have 4, 8 and 12 rate parameters respectively. Construct one combined-character mapping with the same labels, and pass each matrix as `rate_mat` with the same `state_order`. All three comparisons must use the same taxa and observation coding. If one component is unknown, encode all compatible combinations, for example `"00&10"` when the first component is unknown and the second is 0. Both components missing allow all four states.

This is an explicit modeling recipe; the package cannot infer a justified independence or dual-transition hypothesis from category names alone.
