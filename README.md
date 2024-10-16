# Quantile2SP

This repository includes examples for the implementation of "A Quantile Neural Network Framework for Two-stage Stochastic Optimization" (https://arxiv.org/abs/2403.11707)

## Content

- NN_training.py: uses the utils file to train a Quantile Neural Network with and without an Incremental output layer.
- QNN_embedding.py: example of the representation of the QNN as a mixed-integer set of constraints.
- IQNN_embedding.py: example of the representation of the QNN with Incremental output layer as a mixed-integer set of constraints.
- Opt_problems folder: includes a Facility Location Problem example from the paper (extending it for risk-aversion settings), where we embed an IQNN as a surrogate model for the second stage.

## Notes

We thank the authors from Neur2SP (https://github.com/khalil-research/Neur2SP) for providing the original code of the optimization problems. The data in the Opt_problems folder is generated using their repository.

## Reference

Please, cite our paper if you found the ideas or the code useful to your work.

```
@article{alcantara2024quantile,
  title={A Quantile Neural Network Framework for Two-stage Stochastic Optimization},
  author={Alc{\'a}ntara, Antonio and Ruiz, Carlos and Tsay, Calvin},
  journal={arXiv preprint arXiv:2403.11707},
  year={2024}
}
```
