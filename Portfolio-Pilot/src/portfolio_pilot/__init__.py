"""PortfolioPilot: investor profiling and diversified ETF allocation (IE 7275).

Modules
-------
config    settings: seed, column names, split sizes, file locations
utils     run log (Recorder) and survey-weight helpers
prepare   load the raw SCF and ETF files
clean     rule-based cleaning, implicate check, ETF cleaning and returns
split     train / validation / test partitions
features  feature construction (statistics fitted on training data)
target    archetype definition by clustering training households
reduce    principal component analysis
figures   report figures (training data only)
tables    report tables built from the run log
dataset   rebuild the modelling data (M4); the test set is dropped unless use_test is on
models    reference, rule and baseline classifiers (M4)
evaluate  survey-weighted metrics and cross-validation (M4)
plots     M4 figures from the saved validation predictions
run_baselines  one command that writes every M4 result:  python -m portfolio_pilot.run_baselines
"""

__version__ = "0.4.0"
