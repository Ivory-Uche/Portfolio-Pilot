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
"""

__version__ = "0.3.0"
