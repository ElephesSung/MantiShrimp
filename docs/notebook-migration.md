# Orca notebook-function migration

The original notebooks remain unchanged. Their reusable logic has been moved
into importable, testable modules so future notebooks can become thin analysis
documents.

| Notebook responsibility | Package destination |
| --- | --- |
| synthetic homogeneous/heterogeneous rate sampling | `mantishrimp.validation` |
| single-cell and population event simulation | `mantishrimp.validation` |
| four-model SMC inference | `mantishrimp.inference` |
| SMC evidence and pairwise Bayes factors | `mantishrimp.inference`, `mantishrimp.analysis` |
| NetCDF/config/evidence persistence | `mantishrimp.analysis` |
| shared posterior-parameter conversion | `mantishrimp.analysis` |
| scenario ground truths, titles, and safe paths | `mantishrimp.analysis` |
| replicate and sample-size summaries | `mantishrimp.analysis` |
| posterior grids and HDI summaries | `mantishrimp.plotting` |
| BF bands, sweeps, and sample-size trajectories | `mantishrimp.plotting` |
| best-two-model half violins | `mantishrimp.plotting` |

Recognisable Orca names such as `sample_lambda`, `simulate_SingleCell`,
`simulate_Population`, `inference_homo`, `inference_Z2P`, `inference_Dis2P`,
`inference_hetero3`, and `plot_four_model_posteriors` are retained as migration
wrappers or aliases. New code should prefer the snake-case package API.

Notebook-specific global paths, hard-coded scenario lists, and plotting-time
file discovery were intentionally not copied. Callers now pass roots, data
frames, scenarios, and fitted objects explicitly.
