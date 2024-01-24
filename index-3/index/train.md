# 학습

\[TOC]

This library provides a set of classes and functions that helps train models.

## Optimizers

The Optimizer base class provides methods to compute gradients for a loss and apply gradients to variables. A collection of subclasses implement classic optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one of the subclasses.

***

### `class tf.train.Optimizer` <a href="#optimizer" id="optimizer"></a>

Base class for optimizers.

This class defines the API to add Ops to train a model. You never use this class directly, but instead instantiate one of its subclasses such as `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

### Usage

```python
# Create an optimizer with the desired parameters.
opt = GradientDescentOptimizer(learning_rate=0.1)
# Add Ops to the graph to minimize a cost by updating a list of variables.
# "cost" is a Tensor, and the list of variables contains tf.Variable
# objects.
opt_op = opt.minimize(cost, var_list=<list of variables>)
```

In the training program you will just have to run the returned Op.

```python
# Execute opt_op to do one step of training:
opt_op.run()
```

### Processing gradients before applying them.

Calling `minimize()` takes care of both computing the gradients and applying them to the variables. If you want to process the gradients before applying them you can instead use the optimizer in three steps:

1. Compute the gradients with `compute_gradients()`.
2. Process the gradients as you wish.
3. Apply the processed gradients with `apply_gradients()`.

Example:

```python
# Create an optimizer.
opt = GradientDescentOptimizer(learning_rate=0.1)

# Compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(loss, <list of variables>)

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

# Ask the optimizer to apply the capped gradients.
opt.apply_gradients(capped_grads_and_vars)
```

***

#### `tf.train.Optimizer.__init__(use_locking, name)` {#Optimizer.**init**}

Create a new Optimizer.

This must be called by the constructors of subclasses.

**Args:**

* `use_locking`: Bool. If True apply use locks to prevent concurrent updates to variables.
* `name`: A non-empty string. The name to use for accumulators created for the optimizer.

**Raises:**

* `ValueError`: If name is malformed.

***

#### `tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)` <a href="#optimizer.minimize" id="optimizer.minimize"></a>

Add operations to minimize `loss` by updating `var_list`.

This method simply combines calls `compute_gradients()` and `apply_gradients()`. If you want to process the gradient before applying them call `compute_gradients()` and `apply_gradients()` explicitly instead of using this function.

**Args:**

* `loss`: A `Tensor` containing the value to minimize.
* `global_step`: Optional `Variable` to increment by one after the variables have been updated.
* `var_list`: Optional list of `Variable` objects to update to minimize `loss`. Defaults to the list of variables collected in the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
* `gate_gradients`: How to gate the computation of gradients. Can be `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
* `aggregation_method`: Specifies the method used to combine gradient terms. Valid values are defined in the class `AggregationMethod`.
* `colocate_gradients_with_ops`: If True, try colocating gradients with the corresponding op.
* `name`: Optional name for the returned operation.
* `grad_loss`: Optional. A `Tensor` holding the gradient computed for `loss`.

**Returns:**

An Operation that updates the variables in `var_list`. If `global_step` was not `None`, that operation also increments `global_step`.

**Raises:**

* `ValueError`: If some of the variables are not `Variable` objects.

***

#### `tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)` <a href="#optimizer.compute_gradients" id="optimizer.compute_gradients"></a>

Compute gradients of `loss` for the variables in `var_list`.

This is the first part of `minimize()`. It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable". Note that "gradient" can be a `Tensor`, an `IndexedSlices`, or `None` if there is no gradient for the given variable.

**Args:**

* `loss`: A Tensor containing the value to minimize.
* `var_list`: Optional list of tf.Variable to update to minimize `loss`. Defaults to the list of variables collected in the graph under the key `GraphKey.TRAINABLE_VARIABLES`.
* `gate_gradients`: How to gate the computation of gradients. Can be `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
* `aggregation_method`: Specifies the method used to combine gradient terms. Valid values are defined in the class `AggregationMethod`.
* `colocate_gradients_with_ops`: If True, try colocating gradients with the corresponding op.
* `grad_loss`: Optional. A `Tensor` holding the gradient computed for `loss`.

**Returns:**

A list of (gradient, variable) pairs.

**Raises:**

* `TypeError`: If `var_list` contains anything else than `Variable` objects.
* `ValueError`: If some arguments are invalid.

***

#### `tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)` <a href="#optimizer.apply_gradients" id="optimizer.apply_gradients"></a>

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that applies gradients.

**Args:**

* `grads_and_vars`: List of (gradient, variable) pairs as returned by `compute_gradients()`.
* `global_step`: Optional `Variable` to increment by one after the variables have been updated.
* `name`: Optional name for the returned operation. Default to the name passed to the `Optimizer` constructor.

**Returns:**

An `Operation` that applies the specified gradients. If `global_step` was not None, that operation also increments `global_step`.

**Raises:**

* `TypeError`: If `grads_and_vars` is malformed.
* `ValueError`: If none of the variables have gradients.

### Gating Gradients

Both `minimize()` and `compute_gradients()` accept a `gate_gradient` argument that controls the degree of parallelism during the application of the gradients.

The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.

`GATE_NONE`: Compute and apply gradients in parallel. This provides the maximum parallelism in execution, at the cost of some non-reproducibility in the results. For example the two gradients of `matmul` depend on the input values: With `GATE_NONE` one of the gradients could be applied to one of the inputs _before_ the other gradient is computed resulting in non-reproducible results.

`GATE_OP`: For each Op, make sure all gradients are computed before they are used. This prevents race conditions for Ops that generate gradients for multiple inputs where the gradients depend on the inputs.

`GATE_GRAPH`: Make sure all gradients for all variables are computed before any one of them is used. This provides the least parallelism but can be useful if you want to process all gradients before applying any of them.

### Slots

Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer` allocate and manage additional variables associated with the variables to train. These are called _Slots_. Slots have names and you can ask the optimizer for the names of the slots that it uses. Once you have a slot name you can ask the optimizer for the variable it created to hold the slot value.

This can be useful if you want to log debug a training algorithm, report stats about the slots, etc.

***

#### `tf.train.Optimizer.get_slot_names()` <a href="#optimizer.get_slot_names" id="optimizer.get_slot_names"></a>

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

**Returns:**

A list of strings.

***

#### `tf.train.Optimizer.get_slot(var, name)` <a href="#optimizer.get_slot" id="optimizer.get_slot"></a>

Return a slot named `name` created for `var` by the Optimizer.

Some `Optimizer` subclasses use additional variables. For example `Momentum` and `Adagrad` use variables to accumulate updates. This method gives access to these `Variable` objects if for some reason you need them.

Use `get_slot_names()` to get the list of slot names created by the `Optimizer`.

**Args:**

* `var`: A variable passed to `minimize()` or `apply_gradients()`.
* `name`: A string.

**Returns:**

The `Variable` for the slot if it was created, `None` otherwise.

***

### `class tf.train.GradientDescentOptimizer` <a href="#gradientdescentoptimizer" id="gradientdescentoptimizer"></a>

Optimizer that implements the gradient descent algorithm.

***

#### `tf.train.GradientDescentOptimizer.__init__(learning_rate, use_locking=False, name='GradientDescent')` {#GradientDescentOptimizer.**init**}

Construct a new gradient descent optimizer.

**Args:**

* `learning_rate`: A Tensor or a floating point value. The learning rate to use.
* `use_locking`: If True use locks for update operations.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "GradientDescent".

***

### `class tf.train.AdadeltaOptimizer` <a href="#adadeltaoptimizer" id="adadeltaoptimizer"></a>

Optimizer that implements the Adadelta algorithm.

See [M. D. Zeiler](http://arxiv.org/abs/1212.5701) ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))

***

#### `tf.train.AdadeltaOptimizer.__init__(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')` {#AdadeltaOptimizer.**init**}

Construct a new Adadelta optimizer.

**Args:**

* `learning_rate`: A `Tensor` or a floating point value. The learning rate.
* `rho`: A `Tensor` or a floating point value. The decay rate.
* `epsilon`: A `Tensor` or a floating point value. A constant epsilon used to better conditioning the grad update.
* `use_locking`: If `True` use locks for update operations.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "Adadelta".

***

### `class tf.train.AdagradOptimizer` <a href="#adagradoptimizer" id="adagradoptimizer"></a>

Optimizer that implements the Adagrad algorithm.

See this [paper](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).

***

#### `tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')` {#AdagradOptimizer.**init**}

Construct a new Adagrad optimizer.

**Args:**

* `learning_rate`: A `Tensor` or a floating point value. The learning rate.
* `initial_accumulator_value`: A floating point value. Starting value for the accumulators, must be positive.
* `use_locking`: If `True` use locks for update operations.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "Adagrad".

**Raises:**

* `ValueError`: If the `initial_accumulator_value` is invalid.

***

### `class tf.train.MomentumOptimizer` <a href="#momentumoptimizer" id="momentumoptimizer"></a>

Optimizer that implements the Momentum algorithm.

***

#### `tf.train.MomentumOptimizer.__init__(learning_rate, momentum, use_locking=False, name='Momentum')` {#MomentumOptimizer.**init**}

Construct a new Momentum optimizer.

**Args:**

* `learning_rate`: A `Tensor` or a floating point value. The learning rate.
* `momentum`: A `Tensor` or a floating point value. The momentum.
* `use_locking`: If `True` use locks for update operations.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "Momentum".

***

### `class tf.train.AdamOptimizer` <a href="#adamoptimizer" id="adamoptimizer"></a>

Optimizer that implements the Adam algorithm.

See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980) ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).

***

#### `tf.train.AdamOptimizer.__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')` {#AdamOptimizer.**init**}

Construct a new Adam optimizer.

Initialization:

```
m_0 <- 0 (Initialize initial 1st moment vector)
v_0 <- 0 (Initialize initial 2nd moment vector)
t <- 0 (Initialize timestep)
```

The update rule for `variable` with gradient `g` uses an optimization described at the end of section2 of the paper:

```
t <- t + 1
lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

m_t <- beta1 * m_{t-1} + (1 - beta1) * g
v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
```

The default value of 1e-8 for epsilon might not be a good default in general. For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.

**Args:**

* `learning_rate`: A Tensor or a floating point value. The learning rate.
* `beta1`: A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
* `beta2`: A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
* `epsilon`: A small constant for numerical stability.
* `use_locking`: If True use locks for update operations.
* `name`: Optional name for the operations created when applying gradients. Defaults to "Adam".

***

### `class tf.train.FtrlOptimizer` <a href="#ftrloptimizer" id="ftrloptimizer"></a>

Optimizer that implements the FTRL algorithm.

See this [paper](https://www.eecs.tufts.edu/\~dsculley/papers/ad-click-prediction.pdf).

***

#### `tf.train.FtrlOptimizer.__init__(learning_rate, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, use_locking=False, name='Ftrl')` {#FtrlOptimizer.**init**}

Construct a new FTRL optimizer.

**Args:**

* `learning_rate`: A float value or a constant float `Tensor`.
* `learning_rate_power`: A float value, must be less or equal to zero.
* `initial_accumulator_value`: The starting value for accumulators. Only positive values are allowed.
* `l1_regularization_strength`: A float value, must be greater than or equal to zero.
* `l2_regularization_strength`: A float value, must be greater than or equal to zero.
* `use_locking`: If `True` use locks for update operations.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "Ftrl".

**Raises:**

* `ValueError`: If one of the arguments is invalid.

***

### `class tf.train.RMSPropOptimizer` <a href="#rmspropoptimizer" id="rmspropoptimizer"></a>

Optimizer that implements the RMSProp algorithm.

See the \[paper] (http://www.cs.toronto.edu/\~tijmen/csc321/slides/lecture\_slides\_lec6.pdf).

***

#### `tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')` {#RMSPropOptimizer.**init**}

Construct a new RMSProp optimizer.

**Args:**

* `learning_rate`: A Tensor or a floating point value. The learning rate.
* `decay`: Discounting factor for the history/coming gradient
* `momentum`: A scalar tensor.
* `epsilon`: Small value to avoid zero denominator.
* `use_locking`: If True use locks for update operation.
* `name`: Optional name prefix for the operations created when applying gradients. Defaults to "RMSProp".

## Gradient Computation

TensorFlow provides functions to compute the derivatives for a given TensorFlow computation graph, adding operations to the graph. The optimizer classes automatically compute derivatives on your graph, but creators of new Optimizers or expert users can call the lower-level functions below.

***

### `tf.gradients(ys, xs, grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)` <a href="#gradients" id="gradients"></a>

Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`.

`ys` and `xs` are each a `Tensor` or a list of tensors. `grad_ys` is a list of `Tensor`, holding the gradients received by the `ys`. The list must be the same length as `ys`.

`gradients()` adds ops to the graph to output the partial derivatives of `ys` with respect to `xs`. It returns a list of `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)` for y in `ys`.

`grad_ys` is a list of tensors of the same length as `ys` that holds the initial gradients for each y in `ys`. When `grad_ys` is None, we fill in a tensor of '1's of the shape of y for each y in `ys`. A user can provide their own initial `grad_ys` to compute the derivatives using a different initial gradient for each y (e.g., if one wanted to weight the gradient differently for each value in each y).

**Args:**

* `ys`: A `Tensor` or list of tensors to be differentiated.
* `xs`: A `Tensor` or list of tensors to be used for differentiation.
* `grad_ys`: Optional. A `Tensor` or list of tensors the same size as `ys` and holding the gradients computed for each y in `ys`.
* `name`: Optional name to use for grouping all the gradient ops together. defaults to 'gradients'.
* `colocate_gradients_with_ops`: If True, try colocating gradients with the corresponding op.
* `gate_gradients`: If True, add a tuple around the gradients returned for an operations. This avoids some race conditions.
* `aggregation_method`: Specifies the method used to combine gradient terms. Accepted values are constants defined in the class `AggregationMethod`.

**Returns:**

A list of `sum(dy/dx)` for each x in `xs`.

**Raises:**

* `LookupError`: if one of the operations between `x` and `y` does not have a registered gradient function.
* `ValueError`: if the arguments are invalid.

***

### `class tf.AggregationMethod` <a href="#aggregationmethod" id="aggregationmethod"></a>

A class listing aggregation methods used to combine gradients.

Computing partial derivatives can require aggregating gradient contributions. This class lists the various methods that can be used to combine gradients in the graph:

* `ADD_N`: All of the gradient terms are summed as part of one operation using the "AddN" op. It has the property that all gradients must be ready before any aggregation is performed.
* `DEFAULT`: The system-chosen default aggregation method.

***

### `tf.stop_gradient(input, name=None)` <a href="#stop_gradient" id="stop_gradient"></a>

Stops gradient computation.

When executed in a graph, this op outputs its input tensor as-is.

When building ops to compute gradients, this op prevents the contribution of its inputs to be taken into account. Normally, the gradient generator adds ops to a graph to compute the derivatives of a specified 'loss' by recursively finding out inputs that contributed to its computation. If you insert this op in the graph it inputs are masked from the gradient generator. They are not taken into account for computing gradients.

This is useful any time you want to compute a value with TensorFlow but need to pretend that the value was a constant. Some examples include:

* The _EM_ algorithm where the _M-step_ should not involve backpropagation through the output of the _E-step_.
* Contrastive divergence training of Boltzmann machines where, when differentiating the energy function, the training must not backpropagate through the graph that generated the samples from the model.
* Adversarial training, where no backprop should happen through the adversarial example generation process.

**Args:**

* `input`: A `Tensor`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `input`.

## Gradient Clipping

TensorFlow provides several operations that you can use to add clipping functions to your graph. You can use these functions to perform general data clipping, but they're particularly useful for handling exploding or vanishing gradients.

***

### `tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)` <a href="#clip_by_value" id="clip_by_value"></a>

Clips tensor values to a specified min and max.

Given a tensor `t`, this operation returns a tensor of the same type and shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`. Any values less than `clip_value_min` are set to `clip_value_min`. Any values greater than `clip_value_max` are set to `clip_value_max`.

**Args:**

* `t`: A `Tensor`.
* `clip_value_min`: A 0-D (scalar) `Tensor`. The minimum value to clip by.
* `clip_value_max`: A 0-D (scalar) `Tensor`. The maximum value to clip by.
* `name`: A name for the operation (optional).

**Returns:**

A clipped `Tensor`.

***

### `tf.clip_by_norm(t, clip_norm, name=None)` <a href="#clip_by_norm" id="clip_by_norm"></a>

Clips tensor values to a maximum L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation normalizes `t` so that its L2-norm is less than or equal to `clip_norm`. Specifically, if the L2-norm is already less than or equal to `clip_norm`, then `t` is not modified. If the L2-norm is greater than `clip_norm`, then this operation returns a tensor of the same type and shape as `t` with its values set to:

`t * clip_norm / l2norm(t)`

In this case, the L2-norm of the output tensor is `clip_norm`.

This operation is typically used to clip gradients before applying them with an optimizer.

**Args:**

* `t`: A `Tensor`.
* `clip_norm`: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
* `name`: A name for the operation (optional).

**Returns:**

A clipped `Tensor`.

***

### `tf.clip_by_average_norm(t, clip_norm, name=None)` <a href="#clip_by_average_norm" id="clip_by_average_norm"></a>

Clips tensor values to a maximum average L2-norm.

Given a tensor `t`, and a maximum clip value `clip_norm`, this operation normalizes `t` so that its average L2-norm is less than or equal to `clip_norm`. Specifically, if the average L2-norm is already less than or equal to `clip_norm`, then `t` is not modified. If the average L2-norm is greater than `clip_norm`, then this operation returns a tensor of the same type and shape as `t` with its values set to:

`t * clip_norm / l2norm_avg(t)`

In this case, the average L2-norm of the output tensor is `clip_norm`.

This operation is typically used to clip gradients before applying them with an optimizer.

**Args:**

* `t`: A `Tensor`.
* `clip_norm`: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
* `name`: A name for the operation (optional).

**Returns:**

A clipped `Tensor`.

***

### `tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)` <a href="#clip_by_global_norm" id="clip_by_global_norm"></a>

Clips values of multiple tensors by the ratio of the sum of their norms.

Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`, this operation returns a list of clipped tensors `list_clipped` and the global norm (`global_norm`) of all tensors in `t_list`. Optionally, if you've already computed the global norm for `t_list`, you can specify the global norm with `use_norm`.

To perform the clipping, the values `t_list[i]` are set to:

```
t_list[i] * clip_norm / max(global_norm, clip_norm)
```

where:

```
global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
```

If `clip_norm > global_norm` then the entries in `t_list` remain as they are, otherwise they're all shrunk by the global ratio.

Any of the entries of `t_list` that are of type `None` are ignored.

This is the correct way to perform gradient clipping (for example, see [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063) ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

However, it is slower than `clip_by_norm()` because all the parameters must be ready before the clipping operation can be performed.

**Args:**

* `t_list`: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
* `clip_norm`: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
* `use_norm`: A 0-D (scalar) `Tensor` of type `float` (optional). The global norm to use. If not provided, `global_norm()` is used to compute the norm.
* `name`: A name for the operation (optional).

**Returns:**

* `list_clipped`: A list of `Tensors` of the same type as `list_t`.
* `global_norm`: A 0-D (scalar) `Tensor` representing the global norm.

**Raises:**

* `TypeError`: If `t_list` is not a sequence.

***

### `tf.global_norm(t_list, name=None)` <a href="#global_norm" id="global_norm"></a>

Computes the global norm of multiple tensors.

Given a tuple or list of tensors `t_list`, this operation returns the global norm of the elements in all tensors in `t_list`. The global norm is computed as:

`global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

Any entries in `t_list` that are of type None are ignored.

**Args:**

* `t_list`: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
* `name`: A name for the operation (optional).

**Returns:**

A 0-D (scalar) `Tensor` of type `float`.

**Raises:**

* `TypeError`: If `t_list` is not a sequence.

## Decaying the learning rate

***

### `tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)` <a href="#exponential_decay" id="exponential_decay"></a>

Applies exponential decay to the learning rate.

When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies an exponential decay function to a provided initial learning rate. It requires a `global_step` value to compute the decayed learning rate. You can just pass a TensorFlow variable that you increment at each training step.

The function returns the decayed learning rate. It is computed as:

```python
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
```

If the argument `staircase` is `True`, then `global_step /decay_steps` is an integer division and the decayed learning rate follows a staircase function.

Example: decay every 100000 steps with a base of 0.96:

```python
...
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)
```

**Args:**

* `learning_rate`: A scalar `float32` or `float64` `Tensor` or a Python number. The initial learning rate.
* `global_step`: A scalar `int32` or `int64` `Tensor` or a Python number. Global step to use for the decay computation. Must not be negative.
* `decay_steps`: A scalar `int32` or `int64` `Tensor` or a Python number. Must be positive. See the decay computation above.
* `decay_rate`: A scalar `float32` or `float64` `Tensor` or a Python number. The decay rate.
* `staircase`: Boolean. It `True` decay the learning rate at discrete intervals.
* `name`: String. Optional name of the operation. Defaults to 'ExponentialDecay'

**Returns:**

A scalar `Tensor` of the same type as `learning_rate`. The decayed learning rate.

## Moving Averages

Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly.

***

### `class tf.train.ExponentialMovingAverage` <a href="#exponentialmovingaverage" id="exponentialmovingaverage"></a>

Maintains moving averages of variables by employing an exponential decay.

When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations that use averaged parameters sometimes produce significantly better results than the final trained values.

The `apply()` method adds shadow copies of trained variables and add ops that maintain a moving average of the trained variables in their shadow copies. It is used when building the training model. The ops that maintain moving averages are typically run after each training step. The `average()` and `average_name()` methods give access to the shadow variables and their names. They are useful when building an evaluation model, or when restoring a model from a checkpoint file. They help use the moving averages in place of the last trained values for evaluations.

The moving averages are computed using exponential decay. You specify the decay value when creating the `ExponentialMovingAverage` object. The shadow variables are initialized with the same initial values as the trained variables. When you run the ops to maintain the moving averages, each shadow variable is updated with the formula:

`shadow_variable -= (1 - decay) * (shadow_variable - variable)`

This is mathematically equivalent to the classic formula below, but the use of an `assign_sub` op (the `"-="` in the formula) allows concurrent lockless updates to the variables:

`shadow_variable = decay * shadow_variable + (1 - decay) * variable`

Reasonable values for `decay` are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.

Example usage when creating a training model:

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)

...train the model by running training_op...
```

There are two ways to use the moving averages for evaluations:

* Build a model that uses the shadow variables instead of the variables. For this, use the `average()` method which returns the shadow variable for a given variable.
* Build a model normally but load the checkpoint files to evaluate by using the shadow variable names. For this use the `average_name()` method. See the [Saver class](train.md#Saver) for more information on restoring saved variables.

Example of restoring the shadow variable values:

```python
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values
```

***

#### `tf.train.ExponentialMovingAverage.__init__(decay, num_updates=None, name='ExponentialMovingAverage')` {#ExponentialMovingAverage.**init**}

Creates a new ExponentialMovingAverage object.

The `apply()` method has to be called to create shadow variables and add ops to maintain moving averages.

The optional `num_updates` parameter allows one to tweak the decay rate dynamically. . It is typical to pass the count of training steps, usually kept in a variable that is incremented at each step, in which case the decay rate is lower at the start of training. This makes moving averages move faster. If passed, the actual decay rate used is:

`min(decay, (1 + num_updates) / (10 + num_updates))`

**Args:**

* `decay`: Float. The decay to use.
* `num_updates`: Optional count of number of updates applied to variables.
* `name`: String. Optional prefix name to use for the name of ops added in `apply()`.

***

#### `tf.train.ExponentialMovingAverage.apply(var_list=None)` <a href="#exponentialmovingaverage.apply" id="exponentialmovingaverage.apply"></a>

Maintains moving averages of variables.

`var_list` must be a list of `Variable` or `Tensor` objects. This method creates shadow variables for all elements of `var_list`. Shadow variables for `Variable` objects are initialized to the variable's initial value. They will be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection. For `Tensor` objects, the shadow variables are initialized to 0.

shadow variables are created with `trainable=False` and added to the `GraphKeys.ALL_VARIABLES` collection. They will be returned by calls to `tf.all_variables()`.

Returns an op that updates all shadow variables as described above.

Note that `apply()` can be called multiple times with different lists of variables.

**Args:**

* `var_list`: A list of Variable or Tensor objects. The variables and Tensors must be of types float32 or float64.

**Returns:**

An Operation that updates the moving averages.

**Raises:**

* `TypeError`: If the arguments are not all float32 or float64.
* `ValueError`: If the moving average of one of the variables is already being computed.

***

#### `tf.train.ExponentialMovingAverage.average_name(var)` <a href="#exponentialmovingaverage.average_name" id="exponentialmovingaverage.average_name"></a>

Returns the name of the `Variable` holding the average for `var`.

The typical scenario for `ExponentialMovingAverage` is to compute moving averages of variables during training, and restore the variables from the computed moving averages during evaluations.

To restore variables, you have to know the name of the shadow variables. That name and the original variable can then be passed to a `Saver()` object to restore the variable from the moving average value with: `saver = tf.train.Saver({ema.average_name(var): var})`

`average_name()` can be called whether or not `apply()` has been called.

**Args:**

* `var`: A `Variable` object.

**Returns:**

A string: The name of the variable that will be used or was used by the `ExponentialMovingAverage class` to hold the moving average of `var`.

***

#### `tf.train.ExponentialMovingAverage.average(var)` <a href="#exponentialmovingaverage.average" id="exponentialmovingaverage.average"></a>

Returns the `Variable` holding the average of `var`.

**Args:**

* `var`: A `Variable` object.

**Returns:**

A `Variable` object or `None` if the moving average of `var` is not maintained..

***

#### `tf.train.ExponentialMovingAverage.variables_to_restore(moving_avg_variables=None)` <a href="#exponentialmovingaverage.variables_to_restore" id="exponentialmovingaverage.variables_to_restore"></a>

Returns a map of names to `Variables` to restore.

If a variable has a moving average, use the moving average variable name as the restore name; otherwise, use the variable name.

For example,

```python
  variables_to_restore = ema.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)
```

Below is an example of such mapping:

```
  conv/batchnorm/gamma/ExponentialMovingAverage: conv/batchnorm/gamma,
  conv_4/conv2d_params/ExponentialMovingAverage: conv_4/conv2d_params,
  global_step: global_step
```

**Args:**

* `moving_avg_variables`: a list of variables that require to use of the moving variable name to be restored. If None, it will default to variables.moving\_average\_variables() + variables.trainable\_variables()

**Returns:**

A map from restore\_names to variables. The restore\_name can be the moving\_average version of the variable name if it exist, or the original variable name.

## Coordinator and QueueRunner

See [Threading and Queues](../../index-2/index-1.md) for how to use threads and queues. For documentation on the Queue API, see [Queues](io\_ops.md#queues).

***

### `class tf.train.Coordinator` <a href="#coordinator" id="coordinator"></a>

A coordinator for threads.

This class implements a simple mechanism to coordinate the termination of a set of threads.

#### Usage:

```python
# Create a coordinator.
coord = Coordinator()
# Start a number of threads, passing the coordinator to each of them.
...start thread 1...(coord, ...)
...start thread N...(coord, ...)
# Wait for all the threads to terminate.
coord.join(threads)
```

Any of the threads can call `coord.request_stop()` to ask for all the threads to stop. To cooperate with the requests, each thread must check for `coord.should_stop()` on a regular basis. `coord.should_stop()` returns `True` as soon as `coord.request_stop()` has been called.

A typical thread running with a coordinator will do something like:

```python
while not coord.should_stop():
  ...do some work...
```

#### Exception handling:

A thread can report an exception to the coordinator as part of the `should_stop()` call. The exception will be re-raised from the `coord.join()` call.

Thread code:

```python
try:
  while not coord.should_stop():
    ...do some work...
except Exception as e:
  coord.request_stop(e)
```

Main code:

```python
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate.
  coord.join(threads)
except Exception as e:
  ...exception that was passed to coord.request_stop()
```

To simplify the thread implementation, the Coordinator provides a context handler `stop_on_exception()` that automatically requests a stop if an exception is raised. Using the context handler the thread code above can be written as:

```python
with coord.stop_on_exception():
  while not coord.should_stop():
    ...do some work...
```

#### Grace period for stopping:

After a thread has called `coord.request_stop()` the other threads have a fixed time to stop, this is called the 'stop grace period' and defaults to 2 minutes. If any of the threads is still alive after the grace period expires `coord.join()` raises a RuntimeException reporting the laggards.

```python
try:
  ...
  coord = Coordinator()
  # Start a number of threads, passing the coordinator to each of them.
  ...start thread 1...(coord, ...)
  ...start thread N...(coord, ...)
  # Wait for all the threads to terminate, give them 10s grace period
  coord.join(threads, stop_grace_period_secs=10)
except RuntimeException:
  ...one of the threads took more than 10s to stop after request_stop()
  ...was called.
except Exception:
  ...exception that was passed to coord.request_stop()
```

***

#### `tf.train.Coordinator.__init__()` {#Coordinator.**init**}

Create a new Coordinator.

***

#### `tf.train.Coordinator.clear_stop()` <a href="#coordinator.clear_stop" id="coordinator.clear_stop"></a>

Clears the stop flag.

After this is called, calls to `should_stop()` will return `False`.

***

#### `tf.train.Coordinator.join(threads, stop_grace_period_secs=120)` <a href="#coordinator.join" id="coordinator.join"></a>

Wait for threads to terminate.

Blocks until all `threads` have terminated or `request_stop()` is called.

After the threads stop, if an `exc_info` was passed to `request_stop`, that exception is re-raised.

Grace period handling: When `request_stop()` is called, threads are given 'stop\_grace\_period\_secs' seconds to terminate. If any of them is still alive after that period expires, a `RuntimeError` is raised. Note that if an `exc_info` was passed to `request_stop()` then it is raised instead of that `RuntimeError`.

**Args:**

* `threads`: List of `threading.Threads`. The started threads to join.
* `stop_grace_period_secs`: Number of seconds given to threads to stop after `request_stop()` has been called.

**Raises:**

* `RuntimeError`: If any thread is still alive after `request_stop()` is called and the grace period expires.

***

#### `tf.train.Coordinator.request_stop(ex=None)` <a href="#coordinator.request_stop" id="coordinator.request_stop"></a>

Request that the threads stop.

After this is called, calls to `should_stop()` will return `True`.

Note: If an exception is being passed in, in must be in the context of handling the exception (i.e. `try: ... except Exception as ex: ...`) and not a newly created one.

**Args:**

* `ex`: Optional `Exception`, or Python `exc_info` tuple as returned by `sys.exc_info()`. If this is the first call to `request_stop()` the corresponding exception is recorded and re-raised from `join()`.

***

#### `tf.train.Coordinator.should_stop()` <a href="#coordinator.should_stop" id="coordinator.should_stop"></a>

Check if stop was requested.

**Returns:**

True if a stop was requested.

***

#### `tf.train.Coordinator.stop_on_exception()` <a href="#coordinator.stop_on_exception" id="coordinator.stop_on_exception"></a>

Context manager to request stop when an Exception is raised.

Code that uses a coordinator must catch exceptions and pass them to the `request_stop()` method to stop the other threads managed by the coordinator.

This context handler simplifies the exception handling. Use it as follows:

```python
with coord.stop_on_exception():
  # Any exception raised in the body of the with
  # clause is reported to the coordinator before terminating
  # the execution of the body.
  ...body...
```

This is completely equivalent to the slightly longer code:

```python
try:
  ...body...
exception Exception as ex:
  coord.request_stop(ex)
```

**Yields:**

nothing.

***

#### `tf.train.Coordinator.wait_for_stop(timeout=None)` <a href="#coordinator.wait_for_stop" id="coordinator.wait_for_stop"></a>

Wait till the Coordinator is told to stop.

**Args:**

* `timeout`: Float. Sleep for up to that many seconds waiting for should\_stop() to become True.

**Returns:**

True if the Coordinator is told stop, False if the timeout expired.

***

### `class tf.train.QueueRunner` <a href="#queuerunner" id="queuerunner"></a>

Holds a list of enqueue operations for a queue, each to be run in a thread.

Queues are a convenient TensorFlow mechanism to compute tensors asynchronously using multiple threads. For example in the canonical 'Input Reader' setup one set of threads generates filenames in a queue; a second set of threads read records from the files, processes them, and enqueues tensors on a second queue; a third set of threads dequeues these input records to construct batches and runs them through training operations.

There are several delicate issues when running multiple threads that way: closing the queues in sequence as the input is exhausted, correctly catching and reporting exceptions, etc.

The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.

***

#### `tf.train.QueueRunner.__init__(queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_runner_def=None)` {#QueueRunner.**init**}

Create a QueueRunner.

On construction the `QueueRunner` adds an op to close the queue. That op will be run if the enqueue ops raise exceptions.

When you later call the `create_threads()` method, the `QueueRunner` will create one thread for each op in `enqueue_ops`. Each thread will run its enqueue op in parallel with the other threads. The enqueue ops do not have to all be the same op, but it is expected that they all enqueue tensors in `queue`.

**Args:**

* `queue`: A `Queue`.
* `enqueue_ops`: List of enqueue ops to run in threads later.
* `close_op`: Op to close the queue. Pending enqueue ops are preserved.
* `cancel_op`: Op to close the queue and cancel pending enqueue ops.
* `queue_runner_def`: Optional `QueueRunnerDef` protocol buffer. If specified, recreates the QueueRunner from its contents. `queue_runner_def` and the other arguments are mutually exclusive.

**Raises:**

* `ValueError`: If both `queue_runner_def` and `queue` are both specified.
* `ValueError`: If `queue` or `enqueue_ops` are not provided when not restoring from `queue_runner_def`.

***

#### `tf.train.QueueRunner.cancel_op` <a href="#queuerunner.cancel_op" id="queuerunner.cancel_op"></a>

***

#### `tf.train.QueueRunner.close_op` <a href="#queuerunner.close_op" id="queuerunner.close_op"></a>

***

#### `tf.train.QueueRunner.create_threads(sess, coord=None, daemon=False, start=False)` <a href="#queuerunner.create_threads" id="queuerunner.create_threads"></a>

Create threads to run the enqueue ops.

This method requires a session in which the graph was launched. It creates a list of threads, optionally starting them. There is one thread for each op passed in `enqueue_ops`.

The `coord` argument is an optional coordinator, that the threads will use to terminate together and report exceptions. If a coordinator is given, this method starts an additional thread to close the queue when the coordinator requests a stop.

This method may be called again as long as all threads from a previous call have stopped.

**Args:**

* `sess`: A `Session`.
* `coord`: Optional `Coordinator` object for reporting errors and checking stop conditions.
* `daemon`: Boolean. If `True` make the threads daemon threads.
* `start`: Boolean. If `True` starts the threads. If `False` the caller must call the `start()` method of the returned threads.

**Returns:**

A list of threads.

**Raises:**

* `RuntimeError`: If threads from a previous call to `create_threads()` are still running.

***

#### `tf.train.QueueRunner.enqueue_ops` <a href="#queuerunner.enqueue_ops" id="queuerunner.enqueue_ops"></a>

***

#### `tf.train.QueueRunner.exceptions_raised` <a href="#queuerunner.exceptions_raised" id="queuerunner.exceptions_raised"></a>

Exceptions raised but not handled by the `QueueRunner` threads.

Exceptions raised in queue runner threads are handled in one of two ways depending on whether or not a `Coordinator` was passed to `create_threads()`:

* With a `Coordinator`, exceptions are reported to the coordinator and forgotten by the `QueueRunner`.
* Without a `Coordinator`, exceptions are captured by the `QueueRunner` and made available in this `exceptions_raised` property.

**Returns:**

A list of Python `Exception` objects. The list is empty if no exception was captured. (No exceptions are captured when using a Coordinator.)

***

#### `tf.train.QueueRunner.from_proto(queue_runner_def)` <a href="#queuerunner.from_proto" id="queuerunner.from_proto"></a>

Returns a `QueueRunner` object created from `queue_runner_def`.

***

#### `tf.train.QueueRunner.name` <a href="#queuerunner.name" id="queuerunner.name"></a>

The string name of the underlying Queue.

***

#### `tf.train.QueueRunner.queue` <a href="#queuerunner.queue" id="queuerunner.queue"></a>

***

#### `tf.train.QueueRunner.to_proto()` <a href="#queuerunner.to_proto" id="queuerunner.to_proto"></a>

Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

**Returns:**

A `QueueRunnerDef` protocol buffer.

***

### `tf.train.add_queue_runner(qr, collection='queue_runners')` <a href="#add_queue_runner" id="add_queue_runner"></a>

Adds a `QueueRunner` to a collection in the graph.

When building a complex model that uses many queues it is often difficult to gather all the queue runners that need to be run. This convenience function allows you to add a queue runner to a well known collection in the graph.

The companion method `start_queue_runners()` can be used to start threads for all the collected queue runners.

**Args:**

* `qr`: A `QueueRunner`.
* `collection`: A `GraphKey` specifying the graph collection to add the queue runner to. Defaults to `GraphKeys.QUEUE_RUNNERS`.

***

### `tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')` <a href="#start_queue_runners" id="start_queue_runners"></a>

Starts all queue runners collected in the graph.

This is a companion method to `add_queue_runner()`. It just starts threads for all queue runners collected in the graph. It returns the list of all threads.

**Args:**

* `sess`: `Session` used to run the queue ops. Defaults to the default session.
* `coord`: Optional `Coordinator` for coordinating the started threads.
* `daemon`: Whether the threads should be marked as `daemons`, meaning they don't block program exit.
* `start`: Set to `False` to only create the threads, not start them.
* `collection`: A `GraphKey` specifying the graph collection to get the queue runners from. Defaults to `GraphKeys.QUEUE_RUNNERS`.

**Returns:**

A list of threads.

## Distributed execution

See [Distributed TensorFlow](broken-reference) for more information about how to configure a distributed TensorFlow program.

***

### `class tf.train.Server` <a href="#server" id="server"></a>

An in-process TensorFlow server, for use in distributed training.

A `tf.train.Server` instance encapsulates a set of devices and a [`tf.Session`](client.md#Session) target that can participate in distributed training. A server belongs to a cluster (specified by a [`tf.train.ClusterSpec`](train.md#ClusterSpec)), and corresponds to a particular task in a named job. The server can communicate with any other server in the same cluster.

***

#### `tf.train.Server.__init__(server_or_cluster_def, job_name=None, task_index=None, protocol=None, start=True)` {#Server.**init**}

Creates a new server with the given definition.

The `job_name`, `task_index`, and `protocol` arguments are optional, and override any information provided in `server_or_cluster_def`.

**Args:**

* `server_or_cluster_def`: A `tf.train.ServerDef` or `tf.train.ClusterDef` protocol buffer, or a `tf.train.ClusterSpec` object, describing the server to be created and/or the cluster of which it is a member.
* `job_name`: (Optional.) Specifies the name of the job of which the server is a member. Defaults to the value in `server_or_cluster_def`, if specified.
* `task_index`: (Optional.) Specifies the task index of the server in its job. Defaults to the value in `server_or_cluster_def`, if specified. Otherwise defaults to 0 if the server's job has only one task.
* `protocol`: (Optional.) Specifies the protocol to be used by the server. Acceptable values include `"grpc"`. Defaults to the value in `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
* `start`: (Optional.) Boolean, indicating whether to start the server after creating it. Defaults to `True`.

**Raises:**

tf.errors.OpError: Or one of its subclasses if an error occurs while creating the TensorFlow server.

***

#### `tf.train.Server.create_local_server(start=True)` <a href="#server.create_local_server" id="server.create_local_server"></a>

Creates a new single-process cluster running on the local host.

This method is a convenience wrapper for creating a `tf.train.Server` with a `tf.train.ServerDef` that specifies a single-process cluster containing a single task in a job called `"local"`.

**Args:**

* `start`: (Optional.) Boolean, indicating whether to start the server after creating it. Defaults to `True`.

**Returns:**

A local `tf.train.Server`.

***

#### `tf.train.Server.target` <a href="#server.target" id="server.target"></a>

Returns the target for a `tf.Session` to connect to this server.

To create a [`tf.Session`](client.md#Session) that connects to this server, use the following snippet:

```python
server = tf.train.Server(...)
with tf.Session(server.target):
  # ...
```

**Returns:**

A string containing a session target for this server.

***

#### `tf.train.Server.start()` <a href="#server.start" id="server.start"></a>

Starts this server.

**Raises:**

tf.errors.OpError: Or one of its subclasses if an error occurs while starting the TensorFlow server.

***

#### `tf.train.Server.join()` <a href="#server.join" id="server.join"></a>

Blocks until the server has shut down.

This method currently blocks forever.

**Raises:**

tf.errors.OpError: Or one of its subclasses if an error occurs while joining the TensorFlow server.

***

### `class tf.train.Supervisor` <a href="#supervisor" id="supervisor"></a>

A training helper that checkpoints models and computes summaries.

The Supervisor is a small wrapper around a `Coordinator`, a `Saver`, and a `SessionManager` that takes care of common needs of Tensorflow training programs.

#### Use for a single program

```python
with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
  sv = Supervisor(logdir='/tmp/mydir')
  # Get a Tensorflow session managed by the supervisor.
  with sv.managed_session(FLAGS.master) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

Within the `with sv.managed_session()` block all variables in the graph have been initialized. In addition, a few services have been started to checkpoint the model and add summaries to the event log.

If the program crashes and is restarted, the managed session automatically reinitialize variables from the most recent checkpoint.

The supervisor is notified of any exception raised by one of the services. After an exception is raised, `should_stop()` returns `True`. In that case the training loop should also stop. This is why the training loop has to check for `sv.should_stop()`.

Exceptions that indicate that the training inputs have been exhausted, `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True` but are not re-raised from the `with` block: they indicate a normal termination.

#### Use for multiple replicas

To train with replicas you deploy the same program in a `Cluster`. One of the tasks must be identified as the _chief_: the task that handles initialization, checkpoints, summaries, and recovery. The other tasks depend on the _chief_ for these services.

The only change you have to do to the single program code is to indicate if the program is running as the _chief_.

```python
# Choose a task as the chief. This could be based on server_def.task_index,
# or job_def.name, or job_def.tasks. It's entirely up to the end user.
# But there can be only one *chief*.
is_chief = (server_def.task_index == 0)
server = tf.train.Server(server_def)

with tf.Graph().as_default():
  ...add operations to the graph...
  # Create a Supervisor that uses log directory on a shared file system.
  # Indicate if you are the 'chief'
  sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
  # Get a Session in a TensorFlow server on the cluster.
  with sv.managed_session(server.target) as sess:
    # Use the session to train the graph.
    while not sv.should_stop():
      sess.run(<my_train_op>)
```

In the _chief_ task, the `Supervisor` works exactly as in the first example above. In the other tasks `sv.managed_session()` waits for the Model to have been intialized before returning a session to the training code. The non-chief tasks depend on the chief taks for initializing the model.

If one of the tasks crashes and restarts, `managed_session()` checks if the Model is initialized. If yes, it just creates a session and returns it to the training code that proceeds normally. If the model needs to be initialized, the chief task takes care of reinitializing it; the other tasks just wait for the model to have been initialized.

NOTE: This modified program still works fine as a single program. The single program marks itself as the chief.

#### What `master` string to use

Whether you are running on your machine or in the cluster you can use the following values for the --master flag:

* Specifying `''` requests an in-process session that does not use RPC.
* Specifying `'local'` requests a session that uses the RPC-based "Master interface" to run TensorFlow programs. See [`tf.train.Server.create_local_server()`](train.md#Server.create\_local\_server) for details.
* Specifying `'grpc://hostname:port'` requests a session that uses the RPC interface to a specific , and also allows the in-process master to access remote tensorflow workers. Often, it is appropriate to pass `server.target` (for some `tf.train.Server` named \`server).

#### Advanced use

**Launching additional services**

`managed_session()` launches the Checkpoint and Summary services (threads). If you need more services to run you can simply launch them in the block controlled by `managed_session()`.

Example: Start a thread to print losses. We want this thread to run every 60 seconds, so we launch it with `sv.loop()`.

```python
...
sv = Supervisor(logdir='/tmp/mydir')
with sv.managed_session(FLAGS.master) as sess:
  sv.loop(60, print_loss, (sess))
  while not sv.should_stop():
    sess.run(my_train_op)
```

**Launching fewer services**

`managed_session()` launches the "summary" and "checkpoint" threads which use either the optionally `summary_op` and `saver` passed to the constructor, or default ones created automatically by the supervisor. If you want to run your own summary and checkpointing logic, disable these services by passing `None` to the `summary_op` and `saver` parameters.

Example: Create summaries manually every 100 steps in the chief.

```python
# Create a Supervisor with no automatic summaries.
sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
# As summary_op was None, managed_session() does not start the
# summary thread.
with sv.managed_session(FLAGS.master) as sess:
  for step in xrange(1000000):
    if sv.should_stop():
      break
    if is_chief and step % 100 == 0:
      # Create the summary every 100 chief steps.
      sv.summary_computed(sess, sess.run(my_summary_op))
    else:
      # Train normally
      sess.run(my_train_op)
```

**Custom model initialization**

`managed_session()` only supports initializing the model by running an `init_op` or restoring from the latest checkpoint. If you have special initialization needs, see how to specify a `local_init_op` when creating the supervisor. You can also use the `SessionManager` directly to create a session and check if it could be initialized automatically.

***

#### `tf.train.Supervisor.__init__(graph=None, ready_op=0, is_chief=True, init_op=0, init_feed_dict=None, local_init_op=0, logdir=None, summary_op=0, saver=0, global_step=0, save_summaries_secs=120, save_model_secs=600, recovery_wait_secs=30, stop_grace_secs=120, checkpoint_basename='model.ckpt', session_manager=None, summary_writer=0, init_fn=None)` {#Supervisor.**init**}

Create a `Supervisor`.

**Args:**

* `graph`: A `Graph`. The graph that the model will use. Defaults to the default `Graph`. The supervisor may add operations to the graph before creating a session, but the graph should not be modified by the caller after passing it to the supervisor.
* `ready_op`: 1-D string `Tensor`. This tensor is evaluated by supervisors in `prepare_or_wait_for_session()` to check if the model is ready to use. The model is considered ready if it returns an empty array. Defaults to the tensor returned from `tf.report_uninitialized_variables()` If `None`, the model is not checked for readiness.
* `is_chief`: If True, create a chief supervisor in charge of initializing and restoring the model. If False, create a supervisor that relies on a chief supervisor for inits and restore.
* `init_op`: `Operation`. Used by chief supervisors to initialize the model when it can not be recovered. Defaults to an `Operation` that initializes all variables. If `None`, no initialization is done automatically unless you pass a value for `init_fn`, see below.
* `init_feed_dict`: A dictionary that maps `Tensor` objects to feed values. This feed dictionary will be used when `init_op` is evaluated.
* `local_init_op`: `Operation`. Used by all supervisors to run initializations that should run for every new supervisor instance. By default these are table initializers and initializers for local variables. If `None`, no further per supervisor-instance initialization is done automatically.
* `logdir`: A string. Optional path to a directory where to checkpoint the model and log events for the visualizer. Used by chief supervisors. The directory will be created if it does not exist.
* `summary_op`: An `Operation` that returns a Summary for the event logs. Used by chief supervisors if a `logdir` was specified. Defaults to the operation returned from merge\_all\_summaries(). If `None`, summaries are not computed automatically.
* `saver`: A Saver object. Used by chief supervisors if a `logdir` was specified. Defaults to the saved returned by Saver(). If `None`, the model is not saved automatically.
* `global_step`: An integer Tensor of size 1 that counts steps. The value from 'global\_step' is used in summaries and checkpoint filenames. Default to the op named 'global\_step' in the graph if it exists, is of rank 1, size 1, and of type tf.int32 ot tf.int64. If `None` the global step is not recorded in summaries and checkpoint files. Used by chief supervisors if a `logdir` was specified.
* `save_summaries_secs`: Number of seconds between the computation of summaries for the event log. Defaults to 120 seconds. Pass 0 to disable summaries.
* `save_model_secs`: Number of seconds between the creation of model checkpoints. Defaults to 600 seconds. Pass 0 to disable checkpoints.
* `recovery_wait_secs`: Number of seconds between checks that the model is ready. Used by supervisors when waiting for a chief supervisor to initialize or restore the model. Defaults to 30 seconds.
* `stop_grace_secs`: Grace period, in seconds, given to running threads to stop when `stop()` is called. Defaults to 120 seconds.
* `checkpoint_basename`: The basename for checkpoint saving.
* `session_manager`: `SessionManager`, which manages Session creation and recovery. If it is `None`, a default `SessionManager` will be created with the set of arguments passed in for backwards compatibility.
* `summary_writer`: `SummaryWriter` to use or `USE_DEFAULT`. Can be `None` to indicate that no summaries should be written.
* `init_fn`: Optional callable used to initialize the model. Called after the optional `init_op` is called. The callable must accept one argument, the session being initialized.

**Returns:**

A `Supervisor`.

***

#### `tf.train.Supervisor.managed_session(master='', config=None, start_standard_services=True, close_summary_writer=True)` <a href="#supervisor.managed_session" id="supervisor.managed_session"></a>

Returns a context manager for a managed session.

This context manager creates and automatically recovers a session. It optionally starts the standard services that handle checkpoints and summaries. It monitors exceptions raised from the `with` block or from the services and stops the supervisor as needed.

The context manager is typically used as follows:

```python
def train():
  sv = tf.train.Supervisor(...)
  with sv.managed_session(<master>) as sess:
    for step in xrange(..):
      if sv.should_stop():
        break
      sess.run(<my training op>)
      ...do other things needed at each training step...
```

An exception raised from the `with` block or one of the service threads is raised again when the block exits. This is done after stopping all threads and closing the session. For example, an `AbortedError` exception, raised in case of preemption of one of the workers in a distributed model, is raised again when the block exits.

If you want to retry the training loop in case of preemption you can do it as follows:

```python
def main(...):
  while True
    try:
      train()
    except tf.errors.Aborted:
      pass
```

As a special case, exceptions used for control flow, such as `OutOfRangeError` which reports that input queues are exhausted, are not raised again from the `with` block: they indicate a clean termination of the training loop and are considered normal termination.

**Args:**

* `master`: name of the TensorFlow master to use. See the `tf.Session` constructor for how this is interpreted.
* `config`: Optional `ConfigProto` proto used to configure the session. Passed as-is to create the session.
* `start_standard_services`: Whether to start the standard services, such as checkpoint, summary and step counter.
* `close_summary_writer`: Whether to close the summary writer when closing the session. Defaults to True.

**Returns:**

A context manager that yields a `Session` restored from the latest checkpoint or initialized from scratch if not checkpoint exists. The session is closed when the `with` block exits.

***

#### `tf.train.Supervisor.prepare_or_wait_for_session(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` <a href="#supervisor.prepare_or_wait_for_session" id="supervisor.prepare_or_wait_for_session"></a>

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as needed, or wait for a session to be ready. If running as the chief and `start_standard_service` is set to True, also call the session manager to start the standard services.

**Args:**

* `master`: name of the TensorFlow master to use. See the `tf.Session` constructor for how this is interpreted.
* `config`: Optional ConfigProto proto used to configure the session, which is passed as-is to create the session.
* `wait_for_checkpoint`: Whether we should wait for the availability of a checkpoint before creating Session. Defaults to False.
* `max_wait_secs`: Maximum time to wait for the session to become available.
* `start_standard_services`: Whether to start the standard services and the queue runners.

**Returns:**

A Session object that can be used to drive the model.

***

#### `tf.train.Supervisor.start_standard_services(sess)` <a href="#supervisor.start_standard_services" id="supervisor.start_standard_services"></a>

Start the standard services for 'sess'.

This starts services in the background. The services started depend on the parameters to the constructor and may include:

* A Summary thread computing summaries every save\_summaries\_secs.
* A Checkpoint thread saving the model every save\_model\_secs.
* A StepCounter thread measure step time.

**Args:**

* `sess`: A Session.

**Returns:**

A list of threads that are running the standard services. You can use the Supervisor's Coordinator to join these threads with: sv.coord.Join()

**Raises:**

* `RuntimeError`: If called with a non-chief Supervisor.
* `ValueError`: If not `logdir` was passed to the constructor as the services need a log directory.

***

#### `tf.train.Supervisor.start_queue_runners(sess, queue_runners=None)` <a href="#supervisor.start_queue_runners" id="supervisor.start_queue_runners"></a>

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS` are already started automatically when you create a session with the supervisor, so unless you have non-collected queue runners to start you do not need to call this explicitely.

**Args:**

* `sess`: A `Session`.
* `queue_runners`: A list of `QueueRunners`. If not specified, we'll use the list of queue runners gathered in the graph under the key `GraphKeys.QUEUE_RUNNERS`.

**Returns:**

The list of threads started for the `QueueRunners`.

***

#### `tf.train.Supervisor.summary_computed(sess, summary, global_step=None)` <a href="#supervisor.summary_computed" id="supervisor.summary_computed"></a>

Indicate that a summary was computed.

**Args:**

* `sess`: A `Session` object.
* `summary`: A Summary proto, or a string holding a serialized summary proto.
* `global_step`: Int. global step this summary is associated with. If `None`, it will try to fetch the current step.

**Raises:**

* `TypeError`: if 'summary' is not a Summary proto or a string.
* `RuntimeError`: if the Supervisor was created without a `logdir`.

***

#### `tf.train.Supervisor.stop(threads=None, close_summary_writer=True)` <a href="#supervisor.stop" id="supervisor.stop"></a>

Stop the services and the coordinator.

This does not close the session.

**Args:**

* `threads`: Optional list of threads to join with the coordinator. If `None`, defaults to the threads running the standard services, the threads started for `QueueRunners`, and the threads started by the `loop()` method. To wait on additional threads, pass the list in this parameter.
* `close_summary_writer`: Whether to close the `summary_writer`. Defaults to `True` if the summary writer was created by the supervisor, `False` otherwise.

***

#### `tf.train.Supervisor.request_stop(ex=None)` <a href="#supervisor.request_stop" id="supervisor.request_stop"></a>

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

**Args:**

* `ex`: Optional `Exception`, or Python `exc_info` tuple as returned by `sys.exc_info()`. If this is the first call to `request_stop()` the corresponding exception is recorded and re-raised from `join()`.

***

#### `tf.train.Supervisor.should_stop()` <a href="#supervisor.should_stop" id="supervisor.should_stop"></a>

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

**Returns:**

True if the coordinator was told to stop, False otherwise.

***

#### `tf.train.Supervisor.stop_on_exception()` <a href="#supervisor.stop_on_exception" id="supervisor.stop_on_exception"></a>

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

**Returns:**

A context handler.

***

#### `tf.train.Supervisor.wait_for_stop()` <a href="#supervisor.wait_for_stop" id="supervisor.wait_for_stop"></a>

Block waiting for the coordinator to stop.

#### Other Methods

***

#### `tf.train.Supervisor.Loop(timer_interval_secs, target, args=None, kwargs=None)` <a href="#supervisor.loop" id="supervisor.loop"></a>

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)` repeatedly. Otherwise it calls it every `timer_interval_secs` seconds. The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor so it does not need to be passed to the `stop()` method.

**Args:**

* `timer_interval_secs`: Number. Time boundaries at which to call `target`.
* `target`: A callable object.
* `args`: Optional arguments to pass to `target` when calling it.
* `kwargs`: Optional keyword arguments to pass to `target` when calling it.

**Returns:**

The started thread.

***

#### `tf.train.Supervisor.PrepareSession(master='', config=None, wait_for_checkpoint=False, max_wait_secs=7200, start_standard_services=True)` <a href="#supervisor.preparesession" id="supervisor.preparesession"></a>

Make sure the model is ready to be used.

Create a session on 'master', recovering or initializing the model as needed, or wait for a session to be ready. If running as the chief and `start_standard_service` is set to True, also call the session manager to start the standard services.

**Args:**

* `master`: name of the TensorFlow master to use. See the `tf.Session` constructor for how this is interpreted.
* `config`: Optional ConfigProto proto used to configure the session, which is passed as-is to create the session.
* `wait_for_checkpoint`: Whether we should wait for the availability of a checkpoint before creating Session. Defaults to False.
* `max_wait_secs`: Maximum time to wait for the session to become available.
* `start_standard_services`: Whether to start the standard services and the queue runners.

**Returns:**

A Session object that can be used to drive the model.

***

#### `tf.train.Supervisor.RequestStop(ex=None)` <a href="#supervisor.requeststop" id="supervisor.requeststop"></a>

Request that the coordinator stop the threads.

See `Coordinator.request_stop()`.

**Args:**

* `ex`: Optional `Exception`, or Python `exc_info` tuple as returned by `sys.exc_info()`. If this is the first call to `request_stop()` the corresponding exception is recorded and re-raised from `join()`.

***

#### `tf.train.Supervisor.ShouldStop()` <a href="#supervisor.shouldstop" id="supervisor.shouldstop"></a>

Check if the coordinator was told to stop.

See `Coordinator.should_stop()`.

**Returns:**

True if the coordinator was told to stop, False otherwise.

***

#### `tf.train.Supervisor.StartQueueRunners(sess, queue_runners=None)` <a href="#supervisor.startqueuerunners" id="supervisor.startqueuerunners"></a>

Start threads for `QueueRunners`.

Note that the queue runners collected in the graph key `QUEUE_RUNNERS` are already started automatically when you create a session with the supervisor, so unless you have non-collected queue runners to start you do not need to call this explicitely.

**Args:**

* `sess`: A `Session`.
* `queue_runners`: A list of `QueueRunners`. If not specified, we'll use the list of queue runners gathered in the graph under the key `GraphKeys.QUEUE_RUNNERS`.

**Returns:**

The list of threads started for the `QueueRunners`.

***

#### `tf.train.Supervisor.StartStandardServices(sess)` <a href="#supervisor.startstandardservices" id="supervisor.startstandardservices"></a>

Start the standard services for 'sess'.

This starts services in the background. The services started depend on the parameters to the constructor and may include:

* A Summary thread computing summaries every save\_summaries\_secs.
* A Checkpoint thread saving the model every save\_model\_secs.
* A StepCounter thread measure step time.

**Args:**

* `sess`: A Session.

**Returns:**

A list of threads that are running the standard services. You can use the Supervisor's Coordinator to join these threads with: sv.coord.Join()

**Raises:**

* `RuntimeError`: If called with a non-chief Supervisor.
* `ValueError`: If not `logdir` was passed to the constructor as the services need a log directory.

***

#### `tf.train.Supervisor.Stop(threads=None, close_summary_writer=True)` <a href="#supervisor.stop" id="supervisor.stop"></a>

Stop the services and the coordinator.

This does not close the session.

**Args:**

* `threads`: Optional list of threads to join with the coordinator. If `None`, defaults to the threads running the standard services, the threads started for `QueueRunners`, and the threads started by the `loop()` method. To wait on additional threads, pass the list in this parameter.
* `close_summary_writer`: Whether to close the `summary_writer`. Defaults to `True` if the summary writer was created by the supervisor, `False` otherwise.

***

#### `tf.train.Supervisor.StopOnException()` <a href="#supervisor.stoponexception" id="supervisor.stoponexception"></a>

Context handler to stop the supervisor when an exception is raised.

See `Coordinator.stop_on_exception()`.

**Returns:**

A context handler.

***

#### `tf.train.Supervisor.SummaryComputed(sess, summary, global_step=None)` <a href="#supervisor.summarycomputed" id="supervisor.summarycomputed"></a>

Indicate that a summary was computed.

**Args:**

* `sess`: A `Session` object.
* `summary`: A Summary proto, or a string holding a serialized summary proto.
* `global_step`: Int. global step this summary is associated with. If `None`, it will try to fetch the current step.

**Raises:**

* `TypeError`: if 'summary' is not a Summary proto or a string.
* `RuntimeError`: if the Supervisor was created without a `logdir`.

***

#### `tf.train.Supervisor.WaitForStop()` <a href="#supervisor.waitforstop" id="supervisor.waitforstop"></a>

Block waiting for the coordinator to stop.

***

#### `tf.train.Supervisor.coord` <a href="#supervisor.coord" id="supervisor.coord"></a>

Return the Coordinator used by the Supervisor.

The Coordinator can be useful if you want to run multiple threads during your training.

**Returns:**

A Coordinator object.

***

#### `tf.train.Supervisor.global_step` <a href="#supervisor.global_step" id="supervisor.global_step"></a>

Return the global\_step Tensor used by the supervisor.

**Returns:**

An integer Tensor for the global\_step.

***

#### `tf.train.Supervisor.init_feed_dict` <a href="#supervisor.init_feed_dict" id="supervisor.init_feed_dict"></a>

Return the feed dictionary used when evaluating the `init_op`.

**Returns:**

A feed dictionary or `None`.

***

#### `tf.train.Supervisor.init_op` <a href="#supervisor.init_op" id="supervisor.init_op"></a>

Return the Init Op used by the supervisor.

**Returns:**

An Op or `None`.

***

#### `tf.train.Supervisor.is_chief` <a href="#supervisor.is_chief" id="supervisor.is_chief"></a>

Return True if this is a chief supervisor.

**Returns:**

A bool.

***

#### `tf.train.Supervisor.loop(timer_interval_secs, target, args=None, kwargs=None)` <a href="#supervisor.loop" id="supervisor.loop"></a>

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)` repeatedly. Otherwise it calls it every `timer_interval_secs` seconds. The thread terminates when a stop is requested.

The started thread is added to the list of threads managed by the supervisor so it does not need to be passed to the `stop()` method.

**Args:**

* `timer_interval_secs`: Number. Time boundaries at which to call `target`.
* `target`: A callable object.
* `args`: Optional arguments to pass to `target` when calling it.
* `kwargs`: Optional keyword arguments to pass to `target` when calling it.

**Returns:**

The started thread.

***

#### `tf.train.Supervisor.ready_op` <a href="#supervisor.ready_op" id="supervisor.ready_op"></a>

Return the Ready Op used by the supervisor.

**Returns:**

An Op or `None`.

***

#### `tf.train.Supervisor.save_model_secs` <a href="#supervisor.save_model_secs" id="supervisor.save_model_secs"></a>

Return the delay between checkpoints.

**Returns:**

A timestamp.

***

#### `tf.train.Supervisor.save_path` <a href="#supervisor.save_path" id="supervisor.save_path"></a>

Return the save path used by the supervisor.

**Returns:**

A string.

***

#### `tf.train.Supervisor.save_summaries_secs` <a href="#supervisor.save_summaries_secs" id="supervisor.save_summaries_secs"></a>

Return the delay between summary computations.

**Returns:**

A timestamp.

***

#### `tf.train.Supervisor.saver` <a href="#supervisor.saver" id="supervisor.saver"></a>

Return the Saver used by the supervisor.

**Returns:**

A Saver object.

***

#### `tf.train.Supervisor.session_manager` <a href="#supervisor.session_manager" id="supervisor.session_manager"></a>

Return the SessionManager used by the Supervisor.

**Returns:**

A SessionManager object.

***

#### `tf.train.Supervisor.summary_op` <a href="#supervisor.summary_op" id="supervisor.summary_op"></a>

Return the Summary Tensor used by the chief supervisor.

**Returns:**

A string Tensor for the summary or `None`.

***

#### `tf.train.Supervisor.summary_writer` <a href="#supervisor.summary_writer" id="supervisor.summary_writer"></a>

Return the SummaryWriter used by the chief supervisor.

**Returns:**

A SummaryWriter.

***

### `class tf.train.SessionManager` <a href="#sessionmanager" id="sessionmanager"></a>

Training helper that restores from checkpoint and creates session.

This class is a small wrapper that takes care of session creation and checkpoint recovery. It also provides functions that to facilitate coordination among multiple training threads or processes.

* Checkpointing trained variables as the training progresses.
* Initializing variables on startup, restoring them from the most recent checkpoint after a crash, or wait for checkpoints to become available.

### Usage:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
  sm = SessionManager()
  sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`prepare_session()` initializes or restores a model. It requires `init_op` and `saver` as an argument.

A second process could wait for the model to be ready by doing the following:

```python
with tf.Graph().as_default():
   ...add operations to the graph...
  # Create a SessionManager that will wait for the model to become ready.
  sm = SessionManager()
  sess = sm.wait_for_session(master)
  # Use the session to train the graph.
  while True:
    sess.run(<my_train_op>)
```

`wait_for_session()` waits for a model to be initialized by other processes.

***

#### `tf.train.SessionManager.__init__(local_init_op=None, ready_op=None, graph=None, recovery_wait_secs=30)` {#SessionManager.**init**}

Creates a SessionManager.

The `local_init_op` is an `Operation` that is run always after a new session was created. If `None`, this step is skipped.

The `ready_op` is an `Operation` used to check if the model is ready. The model is considered ready if that operation returns an empty string tensor. If the operation returns non empty string tensor, the elements are concatenated and used to indicate to the user why the model is not ready.

If `ready_op` is `None`, the model is not checked for readiness.

`recovery_wait_secs` is the number of seconds between checks that the model is ready. It is used by processes to wait for a model to be initialized or restored. Defaults to 30 seconds.

**Args:**

* `local_init_op`: An `Operation` run immediately after session creation. Usually used to initialize tables and local variables.
* `ready_op`: An `Operation` to check if the model is initialized.
* `graph`: The `Graph` that the model will use.
* `recovery_wait_secs`: Seconds between checks for the model to be ready.

***

#### `tf.train.SessionManager.prepare_session(master, init_op=None, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None, init_feed_dict=None, init_fn=None)` <a href="#sessionmanager.prepare_session" id="sessionmanager.prepare_session"></a>

Creates a `Session`. Makes sure the model is ready to be used.

Creates a `Session` on 'master'. If a `saver` object is passed in, and `checkpoint_dir` points to a directory containing valid checkpoint files, then it will try to recover the model from checkpoint. If no checkpoint files are available, and `wait_for_checkpoint` is `True`, then the process would check every `recovery_wait_secs`, up to `max_wait_secs`, for recovery to succeed.

If the model cannot be recovered successfully then it is initialized by either running the provided `init_op`, or calling the provided `init_fn`. It is an error if the model cannot be recovered and neither an `init_op` or an `init_fn` are passed.

This is a convenient function for the following, with a few error checks added:

```python
sess, initialized = self.recover_session(master)
if not initialized:
  if init_op:
    sess.run(init_op, feed_dict=init_feed_dict)
  if init_fn;
    init_fn(sess)
return sess
```

**Args:**

* `master`: `String` representation of the TensorFlow master to use.
* `init_op`: Optional `Operation` used to initialize the model.
* `saver`: A `Saver` object used to restore a model.
* `checkpoint_dir`: Path to the checkpoint files.
* `wait_for_checkpoint`: Whether to wait for checkpoint to become available.
* `max_wait_secs`: Maximum time to wait for checkpoints to become available.
* `config`: Optional `ConfigProto` proto used to configure the session.
* `init_feed_dict`: Optional dictionary that maps `Tensor` objects to feed values. This feed dictionary is passed to the session `run()` call when running the init op.
* `init_fn`: Optional callable used to initialize the model. Called after the optional `init_op` is called. The callable must accept one argument, the session being initialized.

**Returns:**

A `Session` object that can be used to drive the model.

**Raises:**

* `RuntimeError`: If the model cannot be initialized or recovered.

***

#### `tf.train.SessionManager.recover_session(master, saver=None, checkpoint_dir=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None)` <a href="#sessionmanager.recover_session" id="sessionmanager.recover_session"></a>

Creates a `Session`, recovering if possible.

Creates a new session on 'master'. If the session is not initialized and can be recovered from a checkpoint, recover it.

**Args:**

* `master`: `String` representation of the TensorFlow master to use.
* `saver`: A `Saver` object used to restore a model.
* `checkpoint_dir`: Path to the checkpoint files.
* `wait_for_checkpoint`: Whether to wait for checkpoint to become available.
* `max_wait_secs`: Maximum time to wait for checkpoints to become available.
* `config`: Optional `ConfigProto` proto used to configure the session.

**Returns:**

A pair (sess, initialized) where 'initialized' is `True` if the session could be recovered, `False` otherwise.

***

#### `tf.train.SessionManager.wait_for_session(master, config=None, max_wait_secs=inf)` <a href="#sessionmanager.wait_for_session" id="sessionmanager.wait_for_session"></a>

Creates a new `Session` and waits for model to be ready.

Creates a new `Session` on 'master'. Waits for the model to be initialized or recovered from a checkpoint. It's expected that another thread or process will make the model ready, and that this is intended to be used by threads/processes that participate in a distributed training configuration where a different thread/process is responsible for initializing or recovering the model being trained.

NB: The amount of time this method waits for the session is bounded by max\_wait\_secs. By default, this function will wait indefinitely.

**Args:**

* `master`: `String` representation of the TensorFlow master to use.
* `config`: Optional ConfigProto proto used to configure the session.
* `max_wait_secs`: Maximum time to wait for the session to become available.

**Returns:**

A `Session`. May be None if the operation exceeds the timeout specified by config.operation\_timeout\_in\_ms.

**Raises:**

tf.DeadlineExceededError: if the session is not available after max\_wait\_secs.

***

### `class tf.train.ClusterSpec` <a href="#clusterspec" id="clusterspec"></a>

Represents a cluster as a set of "tasks", organized into "jobs".

A `tf.train.ClusterSpec` represents the set of processes that participate in a distributed TensorFlow computation. Every [`tf.train.Server`](train.md#Server) is constructed in a particular cluster.

To create a cluster with two jobs and five tasks, you specify the mapping from job names to lists of network addresses (typically hostname-port pairs).

```
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

***

#### `tf.train.ClusterSpec.as_cluster_def()` <a href="#clusterspec.as_cluster_def" id="clusterspec.as_cluster_def"></a>

Returns a `tf.train.ClusterDef` protocol buffer based on this cluster.

***

#### `tf.train.ClusterSpec.as_dict()` <a href="#clusterspec.as_dict" id="clusterspec.as_dict"></a>

Returns a dictionary from job names to lists of network addresses.

#### Other Methods

***

#### `tf.train.ClusterSpec.__init__(cluster)` {#ClusterSpec.**init**}

Creates a `ClusterSpec`.

**Args:**

* `cluster`: A dictionary mapping one or more job names to lists of network addresses, or a `tf.train.ClusterDef` protocol buffer.

**Raises:**

* `TypeError`: If `cluster` is not a dictionary mapping strings to lists of strings, and not a `tf.train.ClusterDef` protobuf.

***

#### `tf.train.ClusterSpec.job_tasks(job_name)` <a href="#clusterspec.job_tasks" id="clusterspec.job_tasks"></a>

Returns a list of tasks in the given job.

**Args:**

* `job_name`: The string name of a job in this cluster.

**Returns:**

A list of strings, corresponding to the network addresses of tasks in the given job, ordered by task index.

**Raises:**

* `ValueError`: If `job_name` does not name a job in this cluster.

***

#### `tf.train.ClusterSpec.jobs` <a href="#clusterspec.jobs" id="clusterspec.jobs"></a>

Returns a list of job names in this cluster.

**Returns:**

A list of strings, corresponding to the names of jobs in this cluster.

***

### `tf.train.replica_device_setter(ps_tasks=0, ps_device='/job:ps', worker_device='/job:worker', merge_devices=True, cluster=None, ps_ops=None)` <a href="#replica_device_setter" id="replica_device_setter"></a>

Return a `device function` to use when building a Graph for replicas.

Device Functions are used in `with tf.device(device_function):` statement to automatically assign devices to `Operation` objects as they are constructed, Device constraints are added from the inner-most context first, working outwards. The merging behavior adds constraints to fields that are yet unset by a more inner context. Currently the fields are (job, task, cpu/gpu).

If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.

For example,

```python
# To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
# jobs on hosts worker0, worker1 and worker2.
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.replica_device_setter(cluster=cluster_spec)):
  # Build your graph
  v1 = tf.Variable(...)  # assigned to /job:ps/task:0
  v2 = tf.Variable(...)  # assigned to /job:ps/task:1
  v3 = tf.Variable(...)  # assigned to /job:ps/task:0
# Run compute
```

**Args:**

* `ps_tasks`: Number of tasks in the `ps` job.
* `ps_device`: String. Device of the `ps` job. If empty no `ps` job is used. Defaults to `ps`.
* `worker_device`: String. Device of the `worker` job. If empty no `worker` job is used.
* `merge_devices`: `Boolean`. If `True`, merges or only sets a device if the device constraint is completely unset. merges device specification rather than overriding them.
* `cluster`: `ClusterDef` proto or `ClusterSpec`.
* `ps_ops`: List of `Operation` objects that need to be placed on `ps` devices.

**Returns:**

A function to pass to `tf.device()`.

**Raises:**

TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer.

## Summary Operations

The following ops output [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) protocol buffers as serialized string tensors.

You can fetch the output of a summary op in a session, and pass it to a [SummaryWriter](train.md#SummaryWriter) to append it to an event file. Event files contain [`Event`](https://www.tensorflow.org/code/tensorflow/core/util/event.proto) protos that can contain `Summary` protos along with the timestamp and step. You can then use TensorBoard to visualize the contents of the event files. See [TensorBoard and Summaries](../../index-2/index-3.md) for more details.

***

### `tf.scalar_summary(tags, values, collections=None, name=None)` <a href="#scalar_summary" id="scalar_summary"></a>

Outputs a `Summary` protocol buffer with scalar values.

The input `tags` and `values` must have the same shape. The generated summary has a summary value for each tag-value pair in `tags` and `values`.

**Args:**

* `tags`: A `string` `Tensor`. Tags for the summaries.
* `values`: A real numeric Tensor. Values for the summaries.
* `collections`: Optional list of graph collections keys. The new summary op is added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
* `name`: A name for the operation (optional).

**Returns:**

A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.

***

### `tf.image_summary(tag, tensor, max_images=3, collections=None, name=None)` <a href="#image_summary" id="image_summary"></a>

Outputs a `Summary` protocol buffer with images.

The summary has up to `max_images` summary values containing images. The images are built from `tensor` which must be 4-D with shape `[batch_size, height, width, channels]` and where `channels` can be:

* 1: `tensor` is interpreted as Grayscale.
* 3: `tensor` is interpreted as RGB.
* 4: `tensor` is interpreted as RGBA.

The images have the same number of channels as the input tensor. For float input, the values are normalized one image at a time to fit in the range `[0, 255]`. `uint8` values are unchanged. The op uses two different normalization algorithms:

* If the input values are all positive, they are rescaled so the largest one is 255.
* If any input value is negative, the values are shifted so input value 0.0 is at 127. They are then rescaled so that either the smallest value is 0, or the largest one is 255.

The `tag` argument is a scalar `Tensor` of type `string`. It is used to build the `tag` of the summary values:

* If `max_images` is 1, the summary value tag is '_tag_/image'.
* If `max_images` is greater than 1, the summary value tags are generated sequentially as '_tag_/image/0', '_tag_/image/1', etc.

**Args:**

* `tag`: A scalar `Tensor` of type `string`. Used to build the `tag` of the summary values.
* `tensor`: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height, width, channels]` where `channels` is 1, 3, or 4.
* `max_images`: Max number of batch elements to generate images for.
* `collections`: Optional list of ops.GraphKeys. The collections to add the summary to. Defaults to \[ops.GraphKeys.SUMMARIES]
* `name`: A name for the operation (optional).

**Returns:**

A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.

***

### `tf.audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)` <a href="#audio_summary" id="audio_summary"></a>

Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The audio is built from `tensor` which must be 3-D with shape `[batch_size, frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`. It is used to build the `tag` of the summary values:

* If `max_outputs` is 1, the summary value tag is '_tag_/audio'.
* If `max_outputs` is greater than 1, the summary value tags are generated sequentially as '_tag_/audio/0', '_tag_/audio/1', etc.

**Args:**

* `tag`: A scalar `Tensor` of type `string`. Used to build the `tag` of the summary values.
* `tensor`: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]` or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
* `sample_rate`: The sample rate of the signal in hertz.
* `max_outputs`: Max number of batch elements to generate audio for.
* `collections`: Optional list of ops.GraphKeys. The collections to add the summary to. Defaults to \[ops.GraphKeys.SUMMARIES]
* `name`: A name for the operation (optional).

**Returns:**

A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.

***

### `tf.histogram_summary(tag, values, collections=None, name=None)` <a href="#histogram_summary" id="histogram_summary"></a>

Outputs a `Summary` protocol buffer with a histogram.

The generated [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

**Args:**

* `tag`: A `string` `Tensor`. 0-D. Tag to use for the summary value.
* `values`: A real numeric `Tensor`. Any shape. Values to use to build the histogram.
* `collections`: Optional list of graph collections keys. The new summary op is added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
* `name`: A name for the operation (optional).

**Returns:**

A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.

***

### `tf.nn.zero_fraction(value, name=None)` <a href="#zero_fraction" id="zero_fraction"></a>

Returns the fraction of zeros in `value`.

If `value` is empty, the result is `nan`.

This is useful in summaries to measure and report sparsity. For example,

```
z = tf.Relu(...)
summ = tf.scalar_summary('sparsity', tf.nn.zero_fraction(z))
```

**Args:**

* `value`: A tensor of numeric type.
* `name`: A name for the operation (optional).

**Returns:**

The fraction of zeros in `value`, with type `float32`.

***

### `tf.merge_summary(inputs, collections=None, name=None)` <a href="#merge_summary" id="merge_summary"></a>

Merges summaries.

This op creates a [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) protocol buffer that contains the union of all the values in the input summaries.

When the Op is run, it reports an `InvalidArgument` error if multiple values in the summaries to merge use the same tag.

**Args:**

* `inputs`: A list of `string` `Tensor` objects containing serialized `Summary` protocol buffers.
* `collections`: Optional list of graph collections keys. The new summary op is added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
* `name`: A name for the operation (optional).

**Returns:**

A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer resulting from the merging.

***

### `tf.merge_all_summaries(key='summaries')` <a href="#merge_all_summaries" id="merge_all_summaries"></a>

Merges all summaries collected in the default graph.

**Args:**

* `key`: `GraphKey` used to collect the summaries. Defaults to `GraphKeys.SUMMARIES`.

**Returns:**

If no summaries were collected, returns None. Otherwise returns a scalar `Tensor` of type `string` containing the serialized `Summary` protocol buffer resulting from the merging.

## Adding Summaries to Event Files

See [Summaries and TensorBoard](../../index-2/index-3.md) for an overview of summaries, event files, and visualization in TensorBoard.

***

### `class tf.train.SummaryWriter` <a href="#summarywriter" id="summarywriter"></a>

Writes `Summary` protocol buffers to event files.

The `SummaryWriter` class provides a mechanism to create an event file in a given directory and add summaries and events to it. The class updates the file contents asynchronously. This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training.

***

#### `tf.train.SummaryWriter.__init__(logdir, graph=None, max_queue=10, flush_secs=120, graph_def=None)` {#SummaryWriter.**init**}

Creates a `SummaryWriter` and an event file.

On construction the summary writer creates a new event file in `logdir`. This event file will contain `Event` protocol buffers constructed when you call one of the following functions: `add_summary()`, `add_session_log()`, `add_event()`, or `add_graph()`.

If you pass a `Graph` to the constructor it is added to the event file. (This is equivalent to calling `add_graph()` later).

TensorBoard will pick the graph from the file and display it graphically so you can interactively explore the graph you built. You will usually pass the graph from the session in which you launched it:

```python
...create a graph...
# Launch the graph in a session.
sess = tf.Session()
# Create a summary writer, add the 'graph' to the event file.
writer = tf.train.SummaryWriter(<some-directory>, sess.graph)
```

The other arguments to the constructor control the asynchronous writes to the event file:

* `flush_secs`: How often, in seconds, to flush the added summaries and events to disk.
* `max_queue`: Maximum number of summaries or events pending to be written to disk before one of the 'add' calls block.

**Args:**

* `logdir`: A string. Directory where event file will be written.
* `graph`: A `Graph` object, such as `sess.graph`.
* `max_queue`: Integer. Size of the queue for pending events and summaries.
* `flush_secs`: Number. How often, in seconds, to flush the pending events and summaries to disk.
* `graph_def`: DEPRECATED: Use the `graph` argument instead.

***

#### `tf.train.SummaryWriter.add_summary(summary, global_step=None)` <a href="#summarywriter.add_summary" id="summarywriter.add_summary"></a>

Adds a `Summary` protocol buffer to the event file.

This method wraps the provided summary in an `Event` protocol buffer and adds it to the event file.

You can pass the result of evaluating any summary op, using [`Session.run()`](client.md#Session.run) or [`Tensor.eval()`](framework.md#Tensor.eval), to this function. Alternatively, you can pass a `tf.Summary` protocol buffer that you populate with your own data. The latter is commonly done to report evaluation results in event files.

**Args:**

* `summary`: A `Summary` protocol buffer, optionally serialized as a string.
* `global_step`: Number. Optional global step value to record with the summary.

***

#### `tf.train.SummaryWriter.add_session_log(session_log, global_step=None)` <a href="#summarywriter.add_session_log" id="summarywriter.add_session_log"></a>

Adds a `SessionLog` protocol buffer to the event file.

This method wraps the provided session in an `Event` procotol buffer and adds it to the event file.

**Args:**

* `session_log`: A `SessionLog` protocol buffer.
* `global_step`: Number. Optional global step value to record with the summary.

***

#### `tf.train.SummaryWriter.add_event(event)` <a href="#summarywriter.add_event" id="summarywriter.add_event"></a>

Adds an event to the event file.

**Args:**

* `event`: An `Event` protocol buffer.

***

#### `tf.train.SummaryWriter.add_graph(graph, global_step=None, graph_def=None)` <a href="#summarywriter.add_graph" id="summarywriter.add_graph"></a>

Adds a `Graph` to the event file.

The graph described by the protocol buffer will be displayed by TensorBoard. Most users pass a graph in the constructor instead.

**Args:**

* `graph`: A `Graph` object, such as `sess.graph`.
* `global_step`: Number. Optional global step counter to record with the graph.
* `graph_def`: DEPRECATED. Use the `graph` parameter instead.

**Raises:**

* `ValueError`: If both graph and graph\_def are passed to the method.

***

#### `tf.train.SummaryWriter.add_run_metadata(run_metadata, tag, global_step=None)` <a href="#summarywriter.add_run_metadata" id="summarywriter.add_run_metadata"></a>

Adds a metadata information for a single session.run() call.

**Args:**

* `run_metadata`: A `RunMetadata` protobuf object.
* `tag`: The tag name for this metadata.
* `global_step`: Number. Optional global step counter to record with the StepStats.

**Raises:**

* `ValueError`: If the provided tag was already used for this type of event.

***

#### `tf.train.SummaryWriter.flush()` <a href="#summarywriter.flush" id="summarywriter.flush"></a>

Flushes the event file to disk.

Call this method to make sure that all pending events have been written to disk.

***

#### `tf.train.SummaryWriter.close()` <a href="#summarywriter.close" id="summarywriter.close"></a>

Flushes the event file to disk and close the file.

Call this method when you do not need the summary writer anymore.

***

### `tf.train.summary_iterator(path)` <a href="#summary_iterator" id="summary_iterator"></a>

An iterator for reading `Event` protocol buffers from an event file.

You can use this function to read events written to an event file. It returns a Python iterator that yields `Event` protocol buffers.

Example: Print the contents of an events file.

```python
for e in tf.train.summary_iterator(path to events file):
    print(e)
```

Example: Print selected summary values.

```python
# This example supposes that the events file contains summaries with a
# summary value tag 'loss'.  These could have been added by calling
# `add_summary()`, passing the output of a scalar summary op created with
# with: `tf.scalar_summary(['loss'], loss_tensor)`.
for e in tf.train.summary_iterator(path to events file):
    for v in e.summary.value:
        if v.tag == 'loss':
            print(v.simple_value)
```

See the protocol buffer definitions of [Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto) and [Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto) for more information about their attributes.

**Args:**

* `path`: The path to an event file created by a `SummaryWriter`.

**Yields:**

`Event` protocol buffers.

## Training utilities

***

### `tf.train.global_step(sess, global_step_tensor)` <a href="#global_step" id="global_step"></a>

Small helper to get the global step.

```python
# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
# Creates a session.
sess = tf.Session()
# Initializes the variable.
sess.run(global_step_tensor.initializer)
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

global_step: 10
```

**Args:**

* `sess`: A TensorFlow `Session` object.
* `global_step_tensor`: `Tensor` or the `name` of the operation that contains the global step.

**Returns:**

The global step value.

***

### `tf.train.write_graph(graph_def, logdir, name, as_text=True)` <a href="#write_graph" id="write_graph"></a>

Writes a graph proto to a file.

The graph is written as a binary proto unless `as_text` is `True`.

```python
v = tf.Variable(0, name='my_variable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
```

**Args:**

* `graph_def`: A `GraphDef` protocol buffer.
* `logdir`: Directory where to write the graph. This can refer to remote filesystems, such as Google Cloud Storage (GCS).
* `name`: Filename for the graph.
* `as_text`: If `True`, writes the graph as an ASCII proto.

## Other Functions and Classes

***

### `class tf.train.LooperThread` <a href="#looperthread" id="looperthread"></a>

A thread that runs code repeatedly, optionally on a timer.

This thread class is intended to be used with a `Coordinator`. It repeatedly runs code specified either as `target` and `args` or by the `run_loop()` method.

Before each run the thread checks if the coordinator has requested stop. In that case the looper thread terminates immediately.

If the code being run raises an exception, that exception is reported to the coordinator and the thread terminates. The coordinator will then request all the other threads it coordinates to stop.

You typically pass looper threads to the supervisor `Join()` method.

***

#### `tf.train.LooperThread.__init__(coord, timer_interval_secs, target=None, args=None, kwargs=None)` {#LooperThread.**init**}

Create a LooperThread.

**Args:**

* `coord`: A Coordinator.
* `timer_interval_secs`: Time boundaries at which to call Run(), or None if it should be called back to back.
* `target`: Optional callable object that will be executed in the thread.
* `args`: Optional arguments to pass to `target` when calling it.
* `kwargs`: Optional keyword arguments to pass to `target` when calling it.

**Raises:**

* `ValueError`: If one of the arguments is invalid.

***

#### `tf.train.LooperThread.daemon` <a href="#looperthread.daemon" id="looperthread.daemon"></a>

A boolean value indicating whether this thread is a daemon thread (True) or not (False).

This must be set before start() is called, otherwise RuntimeError is raised. Its initial value is inherited from the creating thread; the main thread is not a daemon thread and therefore all threads created in the main thread default to daemon = False.

The entire Python program exits when no alive non-daemon threads are left.

***

#### `tf.train.LooperThread.getName()` <a href="#looperthread.getname" id="looperthread.getname"></a>

***

#### `tf.train.LooperThread.ident` <a href="#looperthread.ident" id="looperthread.ident"></a>

Thread identifier of this thread or None if it has not been started.

This is a nonzero integer. See the thread.get\_ident() function. Thread identifiers may be recycled when a thread exits and another thread is created. The identifier is available even after the thread has exited.

***

#### `tf.train.LooperThread.isAlive()` <a href="#looperthread.isalive" id="looperthread.isalive"></a>

Return whether the thread is alive.

This method returns True just before the run() method starts until just after the run() method terminates. The module function enumerate() returns a list of all alive threads.

***

#### `tf.train.LooperThread.isDaemon()` <a href="#looperthread.isdaemon" id="looperthread.isdaemon"></a>

***

#### `tf.train.LooperThread.is_alive()` <a href="#looperthread.is_alive" id="looperthread.is_alive"></a>

Return whether the thread is alive.

This method returns True just before the run() method starts until just after the run() method terminates. The module function enumerate() returns a list of all alive threads.

***

#### `tf.train.LooperThread.join(timeout=None)` <a href="#looperthread.join" id="looperthread.join"></a>

Wait until the thread terminates.

This blocks the calling thread until the thread whose join() method is called terminates -- either normally or through an unhandled exception or until the optional timeout occurs.

When the timeout argument is present and not None, it should be a floating point number specifying a timeout for the operation in seconds (or fractions thereof). As join() always returns None, you must call isAlive() after join() to decide whether a timeout happened -- if the thread is still alive, the join() call timed out.

When the timeout argument is not present or None, the operation will block until the thread terminates.

A thread can be join()ed many times.

join() raises a RuntimeError if an attempt is made to join the current thread as that would cause a deadlock. It is also an error to join() a thread before it has been started and attempts to do so raises the same exception.

***

#### `tf.train.LooperThread.loop(coord, timer_interval_secs, target, args=None, kwargs=None)` <a href="#looperthread.loop" id="looperthread.loop"></a>

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(args)` repeatedly. Otherwise `target(args)` is called every `timer_interval_secs` seconds. The thread terminates when a stop of the coordinator is requested.

**Args:**

* `coord`: A Coordinator.
* `timer_interval_secs`: Number. Time boundaries at which to call `target`.
* `target`: A callable object.
* `args`: Optional arguments to pass to `target` when calling it.
* `kwargs`: Optional keyword arguments to pass to `target` when calling it.

**Returns:**

The started thread.

***

#### `tf.train.LooperThread.name` <a href="#looperthread.name" id="looperthread.name"></a>

A string used for identification purposes only.

It has no semantics. Multiple threads may be given the same name. The initial name is set by the constructor.

***

#### `tf.train.LooperThread.run()` <a href="#looperthread.run" id="looperthread.run"></a>

***

#### `tf.train.LooperThread.run_loop()` <a href="#looperthread.run_loop" id="looperthread.run_loop"></a>

Called at 'timer\_interval\_secs' boundaries.

***

#### `tf.train.LooperThread.setDaemon(daemonic)` <a href="#looperthread.setdaemon" id="looperthread.setdaemon"></a>

***

#### `tf.train.LooperThread.setName(name)` <a href="#looperthread.setname" id="looperthread.setname"></a>

***

#### `tf.train.LooperThread.start()` <a href="#looperthread.start" id="looperthread.start"></a>

Start the thread's activity.

It must be called at most once per thread object. It arranges for the object's run() method to be invoked in a separate thread of control.

This method will raise a RuntimeError if called more than once on the same thread object.

***

#### `tf.train.LooperThread.start_loop()` <a href="#looperthread.start_loop" id="looperthread.start_loop"></a>

Called when the thread starts.

***

#### `tf.train.LooperThread.stop_loop()` <a href="#looperthread.stop_loop" id="looperthread.stop_loop"></a>

Called when the thread stops.

***

### `tf.train.generate_checkpoint_state_proto(save_dir, model_checkpoint_path, all_model_checkpoint_paths=None)` <a href="#generate_checkpoint_state_proto" id="generate_checkpoint_state_proto"></a>

Generates a checkpoint state proto.

**Args:**

* `save_dir`: Directory where the model was saved.
* `model_checkpoint_path`: The checkpoint file.
* `all_model_checkpoint_paths`: List of strings. Paths to all not-yet-deleted checkpoints, sorted from oldest to newest. If this is a non-empty list, the last element must be equal to model\_checkpoint\_path. These paths are also saved in the CheckpointState proto.

**Returns:**

CheckpointState proto with model\_checkpoint\_path and all\_model\_checkpoint\_paths updated to either absolute paths or relative paths to the current save\_dir.
