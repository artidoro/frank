# FRANK: Factuality Evaluation Benchmark

This repository contains the data for the FRANK Benchmark for factuality evaluation metrics (see our [paper](https://arxiv.org/abs/2104.13346) for more information). The data combines outputs from 9 models on 2 datasets with a total of 2250 annotated model outputs. To have an understanding of the performance of a metric at measuring the factuality of different models on standard summarization datasets we chose to conduct the annotation on recent systems on both CNN/DM and XSum datasets providing a large variety of data and factual errors.

The annotation was conducted based on a typology of factual errors which is described in detail in our paper. Thanks to this fine-grained annotation scheme, we are able to compare specific strength and weaknesses of factuality metrics.

The leaderboard will be made available soon.

## Data
The `data` repository contains the data to run new evaluation metrics and the collected human judgements to compute correlations and anaylsis. All the data comes from the test split of each dataset. We use the hashes from the original datasets to identify the documents.

- `human_annotations.csv` contains one line for each article/model pair. It has a `Factuality` column which is the total human judgement assigned to the summary. This is a score between 0 and 1 as we collected judgements on each sentence and average over sentences. The rest of the columns correspond to individual errors or groups of errors. A 1 indicates that there was no such errors in the summary, a 0 indicates that every sentence contained one such error. We also include columns with "flipped" labels for each category. These columns can be used for the ablation study to determine the influence of each category in the overall correlation with human judgement.
- `benchmark_data.json` is the list of summary/article pairs on which new evaluation metrics have to be executed.
- `selected_documents.txt` is a list of hashes of the documents that were selected to be part of the FRANK benchmark.
- The repositories starting with `articles-...` contain the ouputs of running several evaluation metrics on the benchmark data. These results were used to obtain the correlation numbers in the paper.

## Evaluation
To evaluate new metrics we assess their partial correlation with human judgements. We use partial correlation using the summarization system as control variable. 
The file `evaluate.py` can be used to compute partial correlations along with the Williams test to assess if the difference between two metrics is statistically significant. `evaluate.py` can also be used to conduct a more detailed analysis of the performance of the factuality metric. In particular, it can be used to compute an ablation study and estimate how much the metric is correlated with each category of error. The categories of error are described in the typology defined in our [paper](https://arxiv.org/abs/2104.13346).


The online leaderboard uses the evaluation scripts in `evaluate.py` to evaluate the metrics.

## Submission to the Leaderboard
Submit your metric output using this [Google Form](https://forms.gle/UBC5VCx4t79yjnQ8A). Allow one week to have the results display on the online leaderboard.

### Submission format
We expect a `.json` file like `benchmark_data.json` in the `data` directory with each element having an additional field `score` which will store the score returned by your metric on the corresponding summary/article pair. 

If you have any questions feel free to submit an issue.

## Annotation Tools
Coming soon.

## Citation
```
@article{pagnoni2021understanding,
    title={Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics},
    author={Pagnoni, Artidoro and Balachandran, Vidhisha and Tsvetkov, Yulia},
    journal={arXiv preprint arXiv:2104.13346},
    year={2021}
}
```
