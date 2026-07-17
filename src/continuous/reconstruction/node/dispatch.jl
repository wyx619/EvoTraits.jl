"""
    estim_node(tree, trait, fit; kwargs...)

Perform continuous ancestral state reconstruction (`ASR`) for a previously
fitted continuous model. The accepted keyword arguments depend on the model
family. Regime-aware models accept `edge_segments`; `BMM` also accepts
`mapped_edge` for compatibility.
"""
function estim_node end
