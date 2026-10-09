# Fixed spectral likelihood alternating updates

Read LIKELIHOOD_ALTERNATING_UPDATES_20261009.md for the full Chinese report.

This new experiment differs from the prior 720-graph initialization benchmark: known K, B=C log(n)/n, raw top-absolute-eigenvalue spectra, and no degree vector passed to fitting.

Independent validation: 6 models x 2 sizes x 2 CH strengths x 5 new seeds = 120 graphs. All methods and parameters were frozen before generating seeds 20000--20004. The preferred recipe was a diagnostic follow-up on the earlier 240 graphs, not a prespecified arm of that earlier run.

Run directly from this directory:

    python -m pip install -r requirements.txt
    python independent_validation.py --out independent_new_run --workers 8

The repository includes code, protocols, condition summaries and compact per-graph evaluation records. Full float64 inputs and all iteration/model payloads for the earlier 240 and new 120 graphs are in the separate downloadable ChatGPT attachment spectral_likelihood_updates_20261009.zip (19,087,253 bytes, SHA256 ac78be5097cdbdecef4e60f32afd314f5afd7630a11506cd2c19d0e836241fa9). It is not claimed that those binary archives are committed here. Its saved Library identity is libfile_5f446c1603208191a3125bbc3646fdd0; a subsequent session can resolve it by that ID or exact filename.

After unpacking that complete attachment, run restore_records.py to verify/extract the eight archives. replay.py then fits the preferred method from a saved spectrum without reading original A. The full attachment also includes the earlier variants, convergence traces, restart failures and negative conditions.

Raw A is reproducible from the model, generator and seed. Fitting receives neither A nor ground-truth labels. This is a row-wise working likelihood; exact graph LOO and optimality are not established.
