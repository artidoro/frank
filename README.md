# FRANK: Factuality Evaluation Benchmark

This repository contains the data for the FRANK Benchmark for factuality evaluation metrics (see our NAACL 2021 [paper](https://arxiv.org/abs/2104.13346) for more information). The data combines outputs from 9 models on 2 datasets with a total of 2250 annotated model outputs. We chose to conduct the annotation on recent systems on both CNN/DM and XSum datasets providing a large variety of data and factual errors.

The annotation was conducted based on a typology of factual errors which is described in detail in our paper. Thanks to this fine-grained annotation scheme, the annotations we collected can be used to compare specific strength and weaknesses of factuality metrics.

The leaderboard will be made available soon.

## Data
The `data` repository contains the data to run new evaluation metrics and the collected human judgements to compute correlations and anaylsis. All the data comes from the test split of each dataset. We use the hashes from the original datasets to identify the documents.

- `human_annotations.json` contains one record for each article/model pair. It has a `Factuality` field which is the total human judgement assigned to the summary. This is a score between 0 and 1 as we collected judgements on each sentence and average over sentences. The rest of the fields correspond to individual errors or groups of errors. A 1 indicates that there was no such errors in the summary, a 0 indicates that every sentence contained one such error. We also include fields with "flipped" labels for each category. These fields can be used for the ablation study to determine the influence of each category in the overall correlation with human judgement.
- `benchmark_data.json` is the data on which new evaluation metrics have to be executed. It is a list, each element contains a model-generated summary, a reference summary, and an article. There are additional fields used to track the element: `hash` or identifyier in the dataset the article was taken from, and `model_name` with the name of the model used to generate the summary. 
- `selected_documents.txt` is a list of hashes of the documents that were selected to be part of the FRANK benchmark.
- `baseline_factuality_metrics_outputs.json` contain the ouputs of running several evaluation metrics on the benchmark data. These results were used to obtain the correlation numbers in the paper.

## Evaluation
To evaluate new metrics we assess their partial correlation with human judgements. We use partial correlation using the summarization system as control variable. 
The file `evaluate.py` can be used to compute partial correlations along with the Williams test to assess if the difference between two metrics is statistically significant. `evaluate.py` can also be used to conduct a more detailed analysis of the performance of the factuality metric. In particular, it can be used to compute an ablation study and estimate how much the metric is able to capture each category of error. The categories of error are described in the typology defined in our [paper](https://arxiv.org/abs/2104.13346).

The online leaderboard uses the evaluation scripts in `evaluate.py` to evaluate the metrics.

### Usage
To install requirements:
```python
git clone https://github.com/artidoro/frank.git
cd frank
pip install -r requirements.txt
```

To run on the baseline metrics:
```python
python evaluation/evaluate.py
```

An example submission file is `example_benchmark_data_scored.json`. You can evaluate it with or without baseline metrics using:
```python
python evaluation/evaluate.py --metrics_outputs data/example_benchmark_data_scored.json
python evaluation/evaluate.py --metrics_outputs data/example_benchmark_data_scored.json --no_baseline_metrics
```

If you want to specify how to parse the metric outputs using a json file, you can use the `metrics_outputs_info` argument. `example_metrics_outputs_info.json` is a example file that defines how to parse the `example_benchmark_data_scored.json`. You would use it as follows:
```python
python evaluation/evaluate.py --metrics_outputs_info data/example_metrics_outputs_info.json
```

To use different modes, use the argument `mode`. For example, using the `ablations` mode the script measure how much a metric captures a given type of errors. This is done by computing the negative difference between the partial correlation when flipping the label of one type of error and that without flipping the label. Using the `ablations-plot` generates a plot of the ablations on the selected categories of error.
```python
python evaluation/evaluate.py --mode ablations --metrics_outputs data/example_benchmark_data_scored.json 
```

To compare different metrics, one should test whether their difference is statistically significant. This can be done with the williams test taking into account the metric-metric correlations. Using the mode `mm-correlation` the script computes the the Williams test and the metric-metric correlations.

```python
python evaluation/evaluate.py --mode mm-correlation 
```

One can specify the baseline metrics used for the analysis using the `baseline_metrics` argument. Similarly, the `ablations` arguments specifies which error category to use to compute the ablation study. Note that these can both be changed in the code directly and the other options are commented out for simplicity.


Finally, the code also allows to customize the data split used for the computation of the statistics (dataset and model), the variable used to compute partial correlation, and whether to store the outputs of this tool. See the argument definition for additional help with these options.


## Submission to the Leaderboard
Submit your metric output using this [Google Form](https://forms.gle/UBC5VCx4t79yjnQ8A). Allow one week to have the results display on the online leaderboard.

### Submission format
We expect a `.json` file like `benchmark_data.json` in the `data` directory with each element having an additional field `score` which will store the score returned by your metric on the corresponding summary/article pair. 

You can verify that the `evaluate.py` script works with your `benchmark_data.json`:
```python
    python evaluation/evaluate.py --no_baseline_metrics --metrics_outputs data/example_benchmark_data_scored.json
```

If you have any questions feel free to submit an issue.

## Annotation Tools
Coming soon.

## Citation
```
@inproceedings{pagnoni-2021-frank,
    title={Understanding Factuality in Abstractive Summarization with {FRANK}: A Benchmark for Factuality Metrics},
    author={Pagnoni, Artidoro and Balachandran, Vidhisha and Tsvetkov, Yulia},
    booktitle =   {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
    year =        {2021},
    month =       jun,
    address =     {Mexico City},
}
```
