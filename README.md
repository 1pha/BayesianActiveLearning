# Short Sequence Classification through Active Learning
This code repository is a project for Bayesian Machine Learning course, XAI623 2021 Fall. Detailed usage of the code is explained in [here]("code/../code/README.md).

## Project Description

One advantage of active learning is that we can achieve high performance with less data, by choosing samples that will aid models find the decision boundary. From previous projects, I have classified the paper topics through its title only and achieved 93.8% AUROC with 35k data. Through this project, I would like to discover the followings.
+ Better performance of models with optimization failure, such as classic machine learning models and naive recurrent neural network models.
+ Achieve the performance with less data with transformer-family models.

## Proposed Methods

By controlling **models**, **way of approximation** and **sampling strategy**, I will compare which methods were
### Models
+ Machine Learning Models
+ Recurrent Neural Networks
+ Transformer Models

## Results

## Discussion & Conclusion

## Reference
### Requirements
```text
python==3.7
torch==3.10
wandb==1.12
spacy==3.1
spacy-transformers==1.1.2
transformers==4.5
datasets=
```
### Paper References

*NLP + AL*
+ [Deep Bayesian Active Learning for Natural Language Processing: Results of a Large-Scale Empirical Study](https://arxiv.org/pdf/1808.05697.pdf)
+ [Deep Active Learning for Named Entity Recognition](https://arxiv.org/pdf/1707.05928.pdf)

*Active Learning*
+ Bald paper - [Deep Bayesian Active Learning with Image Data](https://arxiv.org/pdf/1703.02910.pdf)
+ [BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](https://arxiv.org/pdf/1906.08158.pdf)

### Code References
+ [Bayesian Neural Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
+ [BatchBALD](https://github.com/BlackHC/BatchBALD)
+ [Baal](https://github.com/ElementAI/baal)
+ [Active NLP](https://github.com/asiddhant/Active-NLP)
+ [MC Dropout and Model Ensembling](https://github.com/huyng/incertae)
+ [Active Learning](https://github.com/google/active-learning)

### Article Referenecs
+ [Uncertainty Sampling Cheatsheet](https://towardsdatascience.com/uncertainty-sampling-cheatsheet-ec57bc067c0b)
+ [batchbald_redux](https://blackhc.github.io/batchbald_redux/batchbald.html)
+ [Tutorials & Papers about Active Learning](https://github.com/yongjin-shin/awesome-active-learning)