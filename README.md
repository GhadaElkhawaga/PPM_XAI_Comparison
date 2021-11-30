# PPMXIAComparison
This repository contains experiments code of our paper with the title: __"Explainability of Predictive Process Monitoring results: Techniques, Experiments and Lessons Learned"__. Experiments implemented by the code in this repo adhere to the following taxonomy:
![Taxonomy2](https://user-images.githubusercontent.com/48477434/144080560-adfd1f32-c39a-40e9-97e1-d875e203ded4.jpg)

The main file calling other modules in "GeneratingExplanations.py"
- To compare global explanations run "XIA_Comparisons.py"
- To compare local explanations run "XIA_local_Comparisons.py"
- To compare several important features according to several runs of the same perdictive model: "TrialsComparisons.py"
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
