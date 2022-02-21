# PPMXAIComparison
This repository contains experiments code of two papers with the titles:
(1) __"Explainability of Predictive Process Monitoring Results: Can You See My Data Issues?"__. Research conducted in this paper is concerned investigating the effect of different Predictive Process Monitoring (PPM) settings on the data fed into a Machine Learning model and consequently into the employed explainability method. The ultimate goal is to study how differences between resulting explanations indicate several issues in underlying data. 
(2) __"XAI in the context of Predictive Process Monitoring: Too much to Reveal"__. In this paper, a framework is provided to enable studying the effect of different Predictive Process Monitoring (PPM)-related settings and Machine Learning model-related choices on characteristics and expressiveness of resulting explanations. In addition, A comparison is conducted to study how different explainability methods characteristics can shape resulting explanations and enable reflecting underlying model reasoning process

Experiments implemented by the code in this repo adhere to the following taxonomy:

![Taxonomy2](https://user-images.githubusercontent.com/48477434/144080560-adfd1f32-c39a-40e9-97e1-d875e203ded4.jpg)

The main file calling other modules in "GeneratingExplanations.py"
- To compare global explanations run "XAI_Comparisons.py"
- To compare local explanations run "XAI_local_Comparisons.py"
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
