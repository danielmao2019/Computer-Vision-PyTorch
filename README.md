# Computer-Vision-PyTorch <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [1. General info](#1-general-info)
- [2. Setup guide](#2-setup-guide)
- [3. Implementation notes](#3-implementation-notes)
  - [3.1. Variable naming](#31-variable-naming)
  - [3.2. Assumptions for the tasks](#32-assumptions-for-the-tasks)
  - [3.3. When checks are applied](#33-when-checks-are-applied)
- [4. Contributors](#4-contributors)
- [5. See also](#5-see-also)

## 1. General info

This is a code base that aims for production-level implementations of the data input pipeline, definition of popular models, and training and evaluation pipelines. The framework of choice is PyTorch.

I started implementing these in Jan 2023 when working at the Vision and Image Processing Lab at the University of Waterloo, under supervision of Prof. Alexander Wong. Code in this repo are somehow related to my research work but mostly a representation of my understanding of computer vision.

## 2. Setup guide

1. Install `conda` via [this]() link.
2. Create and activate a virtual environment.
3. Install packages from `requirements.txt`.

## 3. Implementation notes

### 3.1. Variable naming

* My convention is to use `size` for the 2-tuple (H, W) and `shape` for the 4-tuple (N, C, H, W).

### 3.2. Assumptions on object type, tensor shape, and tensor data type for different tasks

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

### 3.3. When checks are applied

* For raw datasets using `pytest`.
* Immediately before return in dataloader, as part of program.

### About `softmax`

Classification models, including semantic segmentation models, output pre-softmax class scores.
These scores are converted to probability distribution by the criteria (and metrics) before comparing `y_pred` and `y_true`.

## 4. Contributors

* Daniel Mao, University of Waterloo, [danielmao2019@gmail.com](danielmao2019@gmail.com)

## 5. See also

* Machine-Learning-Knowledge-Base [[GitHub](https://github.com/danielmao2019/Machine-Learning-Knowledge-Base)]
* Computer-Vision-TensorFlow [[GitHub](https://github.com/danielmao2019/Computer-Vision-TensorFlow)]
