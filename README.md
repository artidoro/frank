# FRANK: Factuality Evaluation Benchmark

This repository contains the data for the FRANK Benchmark for factuality evaluation metrics (see our NAACL 2021 [paper](https://arxiv.org/abs/2104.13346) for more information). The data combines outputs from 9 models on 2 datasets with a total of 2250 annotated model outputs. We chose to conduct the annotation on recent systems on both CNN/DM and XSum datasets providing a large variety of data and factual errors.

The annotation was conducted based on a typology of factual errors which is described in detail in our paper. Thanks to this fine-grained annotation scheme, the annotations we collected can be used to compare specific strength and weaknesses of factuality metrics.

The leaderboard website is accessible here https://frank-benchmark.herokuapp.com

## Data
The `data` repository contains the data to run new evaluation metrics and the collected human judgements to compute correlations and anaylsis. All the data comes from the test split of each dataset. We use the hashes from the original datasets to identify the documents.

### Validation-Test Split for FRANK
The FRANK paper presents results on the entire FRANK dataset since the metrics were not tuned for the FRANK benchmark. However, we expect some tuning in future work. For this reason, we split the data in `validation` and `test`. All tuning and experimentation should be performed on the validation set, while the performance results should be reported on the `test` set. The validation set contains summaries from 149 articles (671 summaries) and the test set contains summaries from 350 articles (1575 summaries).

The files all include a `split` field which indicates whether the datapoint is part of the `validation` or `test` set. The fileds `test_split.txt` and `validation_split.txt` contain the list of test and validation hashes. Note that a hash corresponds to an article and all summaries of the same article are in the same split.


### Data description
We describe the contents of the data files below. Note that all files contain a `split` field which indicates whether the datapoint is part of the validation or test split of FRANK.
- `benchmark_data.json` is the data on which new evaluation metrics have to be executed. It is a list, each element contains a model-generated summary, a reference summary, and an article. There are additional fields used to track the element: `hash` or identifyier in the dataset the article was taken from, and `model_name` with the name of the model used to generate the summary. 
- `human_annotations.json` contains one record for each article/model pair. It has a `Factuality` field which is the total human judgement assigned to the summary. This is a score between 0 and 1 as we collected judgements on each sentence and average over sentences. The rest of the fields correspond to individual errors or groups of errors. A 1 indicates that there was no such errors in the summary, a 0 indicates that every sentence contained one such error. We also include fields with "flipped" labels for each category. These fields can be used for the ablation study to determine the influence of each category in the overall correlation with human judgement.
- `human_annotations_sentences.json` contains all human annotations that we collected on the system outputs at the sentence level and for the each annotator (anonymized). We use the same naming convention as in the paper to indicate categories of errors. In addition, "NoE" indicates no error, and "OtherE" indicates an error outside of the typology. This file has the same fields as `human_annotations.json` with two additional fields: `summary_sentence` (the result of running spacy's sentence boundary detection) and `summary_sentences_annotations` which contains the annotations for each sentence. The latter is a list where each element corresponds to a sentence and contains the annoations by the three annoators. Note that an annotator can select more than one category of error if they identify more than one error.
- `selected_documents.txt` is a list of hashes of the documents that were selected to be part of the FRANK benchmark.
- `baseline_factuality_metrics_outputs.json` contain the ouputs of running several evaluation metrics on the benchmark data. These results were used to obtain the correlation numbers in the paper and are helpful to compare new metrics to those previously proposed.


## Evaluation
To evaluate new metrics we assess their partial correlation with human judgements. We use partial correlation using the summarization system as control variable. 
The file `evaluate.py` can be used to compute partial correlations along with the Williams test to assess if the difference between two metrics is statistically significant. `evaluate.py` can also be used to conduct a more detailed analysis of the performance of the factuality metric. In particular, it can be used to compute an ablation study and estimate how much the metric is able to capture each category of error. The categories of error are described in the typology defined in our [paper](https://arxiv.org/abs/2104.13346).

The online leaderboard uses the evaluation scripts in `evaluate.py` to evaluate the metrics.

### Validation-Test Splits for FRANK
We split the data in `validation` and `test`. All tuning and experimentation should be performed on the validation set, while the performance results should be reported on the `test` set. 

### Usage
To install requirements:
```python
git clone https://github.com/artidoro/frank.git
cd frank
pip install -r requirements.txt
```

To run on the baseline metrics on the `validation` set:
```python
python evaluation/evaluate.py
```

To run on the baseline metrics on the `test` set:
```python
python evaluation/evaluate.py --split test
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
The repository with the annotation platoform can be found at https://github.com/artidoro/frank-annotation-platform.

## Citation
```
@inproceedings{pagnoni-etal-2021-understanding,
    title = "Understanding Factuality in Abstractive Summarization with {FRANK}: A Benchmark for Factuality Metrics",
    author = "Pagnoni, Artidoro  and
      Balachandran, Vidhisha  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.383",
    doi = "10.18653/v1/2021.naacl-main.383",
    pages = "4812--4829",
    abstract = "Modern summarization models generate highly fluent but often factually unreliable outputs. This motivated a surge of metrics attempting to measure the factuality of automatically generated summaries. Due to the lack of common benchmarks, these metrics cannot be compared. Moreover, all these methods treat factuality as a binary concept and fail to provide deeper insights on the kinds of inconsistencies made by different systems. To address these limitations, we devise a typology of factual errors and use it to collect human annotations of generated summaries from state-of-the-art summarization systems for the CNN/DM and XSum datasets. Through these annotations we identify the proportion of different categories of factual errors and benchmark factuality metrics, showing their correlation with human judgement as well as their specific strengths and weaknesses.",
}
```
