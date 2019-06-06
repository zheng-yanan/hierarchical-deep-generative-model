## Hierarchical Deep Generative Models with Dual Memory

A TensorFlow implementation of the Hierarchical Deep Generative Models with Dual Memory described in
**Sequence Modeling with Hierarchical Deep Generative Models with Dual Memory**, published as a long paper in CIKM2017.
See the [paper](https://dl.acm.org/citation.cfm?id=3132952) for more details.


## To run the model:
	python trainer.py --dataset <dataset_name> --model <model_name>
will run default training and save model to ./save/hdgm_ptb.

## Prerequisites
 - TensorFlow 1.4.0
 - Python 2.7

## References

If you use any source codes included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

    @inproceedings{DBLP:conf/cikm/ZhengWWYJ17,
      author    = {Yanan Zheng and
                   Lijie Wen and
                   Jianmin Wang and
                   Jun Yan and
                   Lei Ji},
      title     = {Sequence Modeling with Hierarchical Deep Generative Models with Dual
                   Memory},
      booktitle = {{CIKM}},
      pages     = {1369--1378},
      publisher = {{ACM}},
      year      = {2017}
    }
