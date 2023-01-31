# Computer-Vision-PyTorch <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [General info](#general-info)
- [Code base implementation notes](#code-base-implementation-notes)
  - [Assumptions for the tasks](#assumptions-for-the-tasks)
  - [When checks are applied](#when-checks-are-applied)
- [Contributors](#contributors)
- [See also](#see-also)

## General info

This is a code base that aims for production-level implementations of the data input pipeline, definition of popular models, and training and evaluation pipelines. The framework of choice is PyTorch.

I started implementing these in Jan 2023 when working at the Vision and Image Processing Lab at the University of Waterloo, under supervision of Prof. Alexander Wong. Code in this repo are somehow related to my research work but mostly a representation of my understanding of computer vision.

## Code base implementation notes

### Assumptions for the tasks

* Image type: always `torch.Tensor`.
* Image shape: 3-D in raw datasets and 4-D after dataloader due to batching.
* Image dtype: always `torch.float32`.
* Label type: always `torch.Tensor`.
* Label shape:
    * image classification: () in raw datasets and 1-D after dataloader due to batching.
    * object detection: 2-D in raw datasets and 3-D after dataloader due to batching.
    * semantic segmentation: 3-D in raw datasets and 4-D after dataloader due to batching.
* Label dtype:
    * image classification: `torch.int64`
    * object detection: `torch.float32`
    * semantic segmentation: `torch.int64`

### When checks are applied

* For raw datasets using `pytest`.
* Immediately before return in dataloader, as part of program.

## Contributors

* Daniel Mao, University of Waterloo, [danielmao2019@gmail.com](danielmao2019@gmail.com)

## See also

* Machine-Learning-Knowledge-Base [[GitHub](https://github.com/danielmao2019/Machine-Learning-Knowledge-Base)]
* Computer-Vision-TensorFlow [[GitHub](https://github.com/danielmao2019/Computer-Vision-TensorFlow)]
