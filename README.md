# The (Un)Suitability of Automatic Evaluation Metrics for Text Simplification

We present the first meta-evaluation of automatic metrics in Text Simplification to analyse the variation of the correlation between metrics’ scores and human judgments across three dimensions: the perceived simplicity level, the system type, and the set of references used for computation.

![A table showcasing the main findings of our metaevaluation using the Simplicity-DA dataset](https://user-images.githubusercontent.com/2760680/181297857-14cdd48a-0d9d-4bf1-9dee-0a7df505ec81.png)

This repository includes:

- [Simplicity-DA](simplicity_DA.csv), a new dataset with human judgments of simplification quality (in terms of "simplicity") elicited via a methodology inspired by Direct Assessment;
- The individual [ratings_per_annotator](/ratings_per_annotator/) from the Simplicity-DA and the Structural Simplicity datasets;
- [Notebooks](notebooks) with the code to reproduce our meta-evaluation and analyses.

## Licence

Our datasets and scripts are released under a [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence.

## Citation

If you use our resources or findings in your research, please cite our Computational Linguistics article:

> Fernando Alva-Manchego, Carolina Scarton, and Lucia Specia. 2021. [The (Un)Suitability of Automatic Evaluation Metrics for Text Simplification](https://aclanthology.org/2021.cl-4.28/). *Computational Linguistics*, 47(4):861–889.

```BibTeX
@article{alva-manchego-etal-2021-un,
    title = "The (Un)Suitability of Automatic Evaluation Metrics for Text Simplification",
    author = "Alva-Manchego, Fernando  and
      Scarton, Carolina  and
      Specia, Lucia",
    journal = "Computational Linguistics",
    volume = "47",
    number = "4",
    month = dec,
    year = "2021",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2021.cl-4.28",
    doi = "10.1162/coli_a_00418",
    pages = "861--889",
    abstract = "Abstract In order to simplify sentences, several rewriting operations can be performed, such as replacing complex words per simpler synonyms, deleting unnecessary information, and splitting long sentences. Despite this multi-operation nature, evaluation of automatic simplification systems relies on metrics that moderately correlate with human judgments on the simplicity achieved by executing specific operations (e.g., simplicity gain based on lexical replacements). In this article, we investigate how well existing metrics can assess sentence-level simplifications where multiple operations may have been applied and which, therefore, require more general simplicity judgments. For that, we first collect a new and more reliable data set for evaluating the correlation of metrics and human judgments of overall simplicity. Second, we conduct the first meta-evaluation of automatic metrics in Text Simplification, using our new data set (and other existing data) to analyze the variation of the correlation between metrics{'} scores and human judgments across three dimensions: the perceived simplicity level, the system type, and the set of references used for computation. We show that these three aspects affect the correlations and, in particular, highlight the limitations of commonly used operation-specific metrics. Finally, based on our findings, we propose a set of recommendations for automatic evaluation of multi-operation simplifications, suggesting which metrics to compute and how to interpret their scores.",
}
```
