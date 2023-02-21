# Experiment Notes

### General Info

Training batch size is 8 unless otherwise stated.

### LeNet_MNIST_0

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
])
batch_size = 1
```

Training is extremely slow. Probably due to small batch size, longer for loops and less vectorization.

### LeNet_MNIST_Perm_1

```python
criterion = losses.MultiTaskCriterion(criteria=[
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    losses.MappedMNISTCEL(num_classes=10, seed=1),
])
```

Two permuted cross entropy loss, seeded to 0 and 1.

Converged after epoch 25 with train score ~0.05 and eval score start to fluctuate.

Start from epoch 25 saved model and train separately.

### LeNet_MNIST_Perm_5

```python
criterion = losses.MultiTaskCriterion(criteria=[
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    losses.MappedMNISTCEL(num_classes=10, seed=1),
    losses.MappedMNISTCEL(num_classes=10, seed=2),
    losses.MappedMNISTCEL(num_classes=10, seed=3),
    losses.MappedMNISTCEL(num_classes=10, seed=4),
])
```

### LeNet_MNIST_Multi_0

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(mapping='circle'),
    losses.MappedMNISTCEL(mapping='horiz'),
    losses.MappedMNISTCEL(mapping='vert'),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
])
```

### LeNet_MNIST_Multi_1

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(mapping='circle'),
    losses.MappedMNISTCEL(mapping='horiz'),
    losses.MappedMNISTCEL(mapping='vert'),
])
```

### LeNet_MNIST_1

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
])
```

### LeNet_MNIST_Perm_0

```python
criterion = losses.MultiTaskCriterion(criteria=[
    losses.MappedMNISTCEL(num_classes=10, seed=0),
])
```

One permuted cross entropy loss, seeded to 0.

Converged after epoch 35 with train score ~0.1125 and eval score ~0.1135.

### LeNet_MNIST_Multi_2

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[10, 1],
)
```

Trained 200 epochs.

### LeNet_MNIST_Multi_4

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[5, 1],
)
```

Trained 200 epochs.

### LeNet_MNIST_Multi_5

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[4, 1],
)
```

Started.

### LeNet_MNIST_Multi_6

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[3, 1],
)
```

Started.

### LeNet_MNIST_Multi_3

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[1, 1],
)
```

Trained 200 epochs.

### LeNet_MNIST_Multi_3_easy

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[1, 1],
)
```

trained on easy examples.

### LeNet_MNIST_Multi_3_hard

```python
criterion = losses.MultiTaskCriterion(criteria=[
    torch.nn.CrossEntropyLoss(),
    losses.MappedMNISTCEL(num_classes=10, seed=0),
    ], weights=[1, 1],
)
```

trained on hard examples.
