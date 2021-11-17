# PPMXIAComparison
This repository contains experiments code of our paper with the title: __"Explainability of Predictive Process Monitoring results: Techniques, Experiments and Lessons Learned"__
The main file calling other modules in "GeneratingExplanations.py"
- To compare global explanations run "XIA_Comparisons.py"
- To compare local explanations run "XIA_local_Comparisons.py"
- To compare several runs of the same experiments run: "TrialsComparisons.py"
- To compare execution times, run: "timing_comparisons.py"
- For data exploration, run:
  - EDA_and_profiling.py 
  - mutual_info_encoded.py
- To generate general statistics tables and counts, run:
  - DatasetStatisticsGeneration.py
  - correlations_counting.py
  -counts_after_encoding.py
- for necessary data preparations (which should be applied separately), run:
  - encoded_datasets_withLabel.py
  - Discretize.py
