# 제어 연산자

Note: Functions taking `Tensor` arguments can also take anything accepted by [`tf.convert_to_tensor`](broken-reference).

\[TOC]

## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control the execution of operations and add conditional dependencies to your graph.

***

### `tf.identity(input, name=None)` <a href="#identity" id="identity"></a>

Return a tensor with the same shape and contents as the input tensor or value.

**Args:**

* `input`: A `Tensor`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `input`.

***

### `tf.tuple(tensors, name=None, control_inputs=None)` <a href="#tuple" id="tuple"></a>

Group tensors together.

This creates a tuple of tensors with the same values as the `tensors` argument, except that the value of each tensor is only returned after the values of all tensors have been computed.

`control_inputs` contains additional ops that have to finish before this op finishes, but whose outputs are not returned.

This can be used as a "join" mechanism for parallel computations: all the argument tensors can be computed in parallel, but the values of any tensor returned by `tuple` are only available after all the parallel computations are done.

See also `group` and `with_dependencies`.

**Args:**

* `tensors`: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
* `name`: (optional) A name to use as a `name_scope` for the operation.
* `control_inputs`: List of additional ops to finish before returning.

**Returns:**

Same as `tensors`.

**Raises:**

* `ValueError`: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
* `TypeError`: If `control_inputs` is not a list of `Operation` or `Tensor` objects.

***

### `tf.group(*inputs, **kwargs)` <a href="#group" id="group"></a>

Create an op that groups multiple operations.

When this op finishes, all ops in `input` have finished. This op has no output.

See also `tuple` and `with_dependencies`.

**Args:**

* `*inputs`: Zero or more tensors to group.
* `**kwargs`: Optional parameters to pass when constructing the NodeDef.
* `name`: A name for this operation (optional).

**Returns:**

An Operation that executes all its inputs.

**Raises:**

* `ValueError`: If an unknown keyword argument is provided.

***

### `tf.no_op(name=None)` <a href="#no_op" id="no_op"></a>

Does nothing. Only useful as a placeholder for control edges.

**Args:**

* `name`: A name for the operation (optional).

**Returns:**

The created Operation.

***

### `tf.count_up_to(ref, limit, name=None)` <a href="#count_up_to" id="count_up_to"></a>

Increments 'ref' until it reaches 'limit'.

This operation outputs "ref" after the update is done. This makes it easier to chain operations that need to use the updated value.

**Args:**

* `ref`: A mutable `Tensor`. Must be one of the following types: `int32`, `int64`. Should be from a scalar `Variable` node.
* `limit`: An `int`. If incrementing ref would bring it above limit, instead generates an 'OutOfRange' error.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `ref`. A copy of the input before increment. If nothing else modifies the input, the values produced will all be distinct.

***

### `tf.cond(pred, fn1, fn2, name=None)` <a href="#cond" id="cond"></a>

Return either fn1() or fn2() based on the boolean predicate `pred`.

`fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have the same non-zero number and type of outputs.

Note that the conditional execution applies only to the operations defined in fn1 and fn2. Consider the following simple program:

```python
z = tf.mul(a, b)
result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
```

If x < y, the tf.add operation will be executed and tf.square operation will not be executed. Since z is needed for at least one branch of the cond, the tf.mul operation is always executed, unconditionally. Although this behavior is consistent with the dataflow model of TensorFlow, it has occasionally surprised some users who expected a lazier semantics.

**Args:**

* `pred`: A scalar determining whether to return the result of `fn1` or `fn2`.
* `fn1`: The callable to be performed if pred is true.
* `fn2`: The callable to be performed if pref is false.
* `name`: Optional name prefix for the returned tensors.

**Returns:**

Tensors returned by the call to either `fn1` or `fn2`. If the callables return a singleton list, the element is extracted from the list.

**Raises:**

* `TypeError`: if `fn1` or `fn2` is not callable.
* `ValueError`: if `fn1` and `fn2` do not return the same number of tensors, or return tensors of different types.
* `Example`:

```python
  x = tf.constant(2)
  y = tf.constant(5)
  def f1(): return tf.mul(x, 17)
  def f2(): return tf.add(y, 23)
  r = cond(tf.less(x, y), f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
```

***

### `tf.case(pred_fn_pairs, default, exclusive=False, name='case')` <a href="#case" id="case"></a>

Create a case operation.

The `pred_fn_pairs` parameter is a dict or list of pairs of size N. Each pair contains a boolean scalar tensor and a python callable that creates the tensors to be returned if the boolean evaluates to True. `default` is a callable generating a list of tensors. All the callables in `pred_fn_pairs` as well as `default` should return the same number and types of tensors.

If `exclusive==True`, all predicates are evaluated, and a logging operation with an error is returned if more than one of the predicates evaluates to True. If `exclusive==False`, execution stops are the first predicate which evaluates to True, and the tensors generated by the corresponding function are returned immediately. If none of the predicates evaluate to True, this operation returns the tensors generated by `default`.

Example 1: Pseudocode:

```
  if (x < y) return 17;
  else return 23;
```

Expressions:

```
  f1 = lambda: tf.constant(17)
  f2 = lambda: tf.constant(23)
  r = case([(tf.less(x, y), f1)], default=f2)
```

Example 2: Pseudocode:

```
  if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
  if (x < y) return 17;
  else if (x > z) return 23;
  else return -1;
```

Expressions:

```
  x = tf.constant(0)
  y = tf.constant(1)
  z = tf.constant(2)
  def f1(): return tf.constant(17)
  def f2(): return tf.constant(23)
  def f3(): return tf.constant(-1)
  r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
           default=f3, exclusive=True)
```

**Args:**

* `pred_fn_pairs`: Dict or list of pairs of a boolean scalar tensor and a callable which returns a list of tensors.
* `default`: A callable that returns a list of tensors.
* `exclusive`: True iff more than one predicate is allowed to evaluate to True.
* `name`: A name for this operation (optional).

**Returns:**

The tensors returned by the first pair whose predicate evaluated to True, or those returned by `default` if none does.

**Raises:**

* `TypeError`: If `pred_fn_pairs` is not a list/dictionary.
* `TypeError`: If `pred_fn_pairs` is a list but does not contain 2-tuples.
* `TypeError`: If `fns[i]` is not callable for any i, or `default` is not callable.

***

### `tf.while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None)` <a href="#while_loop" id="while_loop"></a>

Repeat `body` while the condition `cond` is true.

`cond` is a callable returning a boolean scalar tensor. `body` is a callable returning a list of tensors of the same length and with the same types as `loop_vars`. `loop_vars` is a list of tensors that is passed to both `cond` and `body`. `cond` and `body` both take as many arguments as there are `loop_vars`.

In addition to regular Tensors or IndexedSlices, the body may accept and return TensorArray objects. The flows of the TensorArray objects will be appropriately forwarded between loops and during gradient calculations.

While `cond` evaluates to true, `body` is executed.

`while_loop` implements non-strict semantics, enabling multiple iterations to run in parallel. The maximum number of parallel iterations can be controlled by `parallel_iterations`, which gives users some control over memory consumption and execution order. For correct programs, `while_loop` should return the same result for any parallel\_iterations > 0.

For training, TensorFlow remembers the tensors that are produced in the forward inference but needed in back propagation. These tensors can be a main source of memory consumption and often cause OOM problems when training on GPUs. When the flag swap\_memory is true, we swap out these tensors from GPU to CPU. This for example allows us to train RNN models with very long sequences and large batches.

**Args:**

* `cond`: A callable that represents the termination condition of the loop.
* `body`: A callable that represents the loop body.
* `loop_vars`: The list of variable input tensors.
* `parallel_iterations`: The number of iterations allowed to run in parallel.
* `back_prop`: Whether backprop is enabled for this while loop.
* `swap_memory`: Whether GPU-CPU memory swap is enabled for this loop.
* `name`: Optional name prefix for the returned tensors.

**Returns:**

The output tensors for the loop variables after the loop.

**Raises:**

* `TypeError`: if `cond` or `body` is not callable.
* `ValueError`: if `loop_var` is empty.
* `Example`:

```python
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
```

## Logical Operators

TensorFlow provides several operations that you can use to add logical operators to your graph.

***

### `tf.logical_and(x, y, name=None)` <a href="#logical_and" id="logical_and"></a>

Returns the truth value of x AND y element-wise.

**Args:**

* `x`: A `Tensor` of type `bool`.
* `y`: A `Tensor` of type `bool`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.logical_not(x, name=None)` <a href="#logical_not" id="logical_not"></a>

Returns the truth value of NOT x element-wise.

**Args:**

* `x`: A `Tensor` of type `bool`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.logical_or(x, y, name=None)` <a href="#logical_or" id="logical_or"></a>

Returns the truth value of x OR y element-wise.

**Args:**

* `x`: A `Tensor` of type `bool`.
* `y`: A `Tensor` of type `bool`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.logical_xor(x, y, name='LogicalXor')` <a href="#logical_xor" id="logical_xor"></a>

x ^ y = (x | y) & \~(x & y).

## Comparison Operators

TensorFlow provides several operations that you can use to add comparison operators to your graph.

***

### `tf.equal(x, y, name=None)` <a href="#equal" id="equal"></a>

Returns the truth value of (x == y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.not_equal(x, y, name=None)` <a href="#not_equal" id="not_equal"></a>

Returns the truth value of (x != y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.less(x, y, name=None)` <a href="#less" id="less"></a>

Returns the truth value of (x < y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.less_equal(x, y, name=None)` <a href="#less_equal" id="less_equal"></a>

Returns the truth value of (x <= y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.greater(x, y, name=None)` <a href="#greater" id="greater"></a>

Returns the truth value of (x > y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.greater_equal(x, y, name=None)` <a href="#greater_equal" id="greater_equal"></a>

Returns the truth value of (x >= y) element-wise.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
* `y`: A `Tensor`. Must have the same type as `x`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.select(condition, t, e, name=None)` <a href="#select" id="select"></a>

Selects elements from `t` or `e`, depending on `condition`.

The `t`, and `e` tensors must all have the same shape, and the output will also have that shape. The `condition` tensor must be a scalar if `t` and `e` are scalars. If `t` and `e` are vectors or higher rank, then `condition` must be either a vector with size matching the first dimension of `t`, or must have the same shape as `t`.

The `condition` tensor acts as a mask that chooses, based on the value at each element, whether the corresponding element / row in the output should be taken from `t` (if true) or `e` (if false).

If `condition` is a vector and `t` and `e` are higher rank matrices, then it chooses which row (outer dimension) to copy from `t` and `e`. If `condition` has the same shape as `t` and `e`, then it chooses which element to copy from `t` and `e`.

For example:

```prettyprint
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 6],
                             [7, 4]]


# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]

```

**Args:**

* `condition`: A `Tensor` of type `bool`.
* `t`: A `Tensor` which may have the same shape as `condition`. If `condition` is rank 1, `t` may have higher rank, but its first dimension must match the size of `condition`.
* `e`: A `Tensor` with the same type and shape as `t`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` with the same type and shape as `t` and `e`.

***

### `tf.where(input, name=None)` <a href="#where" id="where"></a>

Returns locations of true values in a boolean tensor.

This operation returns the coordinates of true elements in `input`. The coordinates are returned in a 2-D tensor where the first dimension (rows) represents the number of true elements, and the second dimension (columns) represents the coordinates of the true elements. Keep in mind, the shape of the output tensor can vary depending on how many true values there are in `input`. Indices are output in row-major order.

For example:

```prettyprint
# 'input' tensor is [[True, False]
#                    [True, False]]
# 'input' has two true values, so output has two coordinates.
# 'input' has rank of 2, so coordinates have two indices.
where(input) ==> [[0, 0],
                  [1, 0]]

# `input` tensor is [[[True, False]
#                     [True, False]]
#                    [[False, True]
#                     [False, True]]
#                    [[False, False]
#                     [False, True]]]
# 'input' has 5 true values, so output has 5 coordinates.
# 'input' has rank of 3, so coordinates have three indices.
where(input) ==> [[0, 0, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [1, 1, 1],
                  [2, 1, 1]]
```

**Args:**

* `input`: A `Tensor` of type `bool`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `int64`.

## Debugging Operations

TensorFlow provides several operations that you can use to validate values and debug your graph.

***

### `tf.is_finite(x, name=None)` <a href="#is_finite" id="is_finite"></a>

Returns which elements of x are finite.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.is_inf(x, name=None)` <a href="#is_inf" id="is_inf"></a>

Returns which elements of x are Inf.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.is_nan(x, name=None)` <a href="#is_nan" id="is_nan"></a>

Returns which elements of x are NaN.

**Args:**

* `x`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor` of type `bool`.

***

### `tf.verify_tensor_all_finite(t, msg, name=None)` <a href="#verify_tensor_all_finite" id="verify_tensor_all_finite"></a>

Assert that the tensor does not contain any NaN's or Inf's.

**Args:**

* `t`: Tensor to check.
* `msg`: Message to log on failure.
* `name`: A name for this operation (optional).

**Returns:**

Same tensor as `t`.

***

### `tf.check_numerics(tensor, message, name=None)` <a href="#check_numerics" id="check_numerics"></a>

Checks a tensor for NaN and Inf values.

When run, reports an `InvalidArgument` error if `tensor` has any values that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

**Args:**

* `tensor`: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
* `message`: A `string`. Prefix of the error message.
* `name`: A name for the operation (optional).

**Returns:**

A `Tensor`. Has the same type as `tensor`.

***

### `tf.add_check_numerics_ops()` <a href="#add_check_numerics_ops" id="add_check_numerics_ops"></a>

Connect a `check_numerics` to every floating point tensor.

`check_numerics` operations themselves are added for each `float` or `double` tensor in the graph. For all ops in the graph, the `check_numerics` op for all of its (`float` or `double`) inputs is guaranteed to run before the `check_numerics` op on any of its outputs.

**Returns:**

A `group` op depending on all `check_numerics` ops added.

***

### `tf.Assert(condition, data, summarize=None, name=None)` <a href="#assert" id="assert"></a>

Asserts that the given condition is true.

If `condition` evaluates to false, print the list of tensors in `data`. `summarize` determines how many entries of the tensors to print.

NOTE: To ensure that Assert executes, one usually attaches a dependency:

```python
 # Ensure maximum element of x is smaller or equal to 1
assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
x = tf.with_dependencies([assert_op], x)
```

**Args:**

* `condition`: The condition to evaluate.
* `data`: The tensors to print out when condition is false.
* `summarize`: Print this many entries of each tensor.
* `name`: A name for this operation (optional).

**Returns:**

* `assert_op`: An `Operation` that, when executed, raises a `tf.errors.InvalidArgumentError` if `condition` is not true.

***

### `tf.Print(input_, data, message=None, first_n=None, summarize=None, name=None)` <a href="#print" id="print"></a>

Prints a list of tensors.

This is an identity op with the side effect of printing `data` when evaluating.

**Args:**

* `input_`: A tensor passed through this op.
* `data`: A list of tensors to print out when op is evaluated.
* `message`: A string, prefix of the error message.
* `first_n`: Only log `first_n` number of times. Negative numbers log always; this is the default.
* `summarize`: Only print this many entries of each tensor. If None, then a maximum of 3 elements are printed per input tensor.
* `name`: A name for the operation (optional).

**Returns:**

Same tensor as `input_`.
