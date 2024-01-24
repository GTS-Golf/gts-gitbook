# 그래프 생성

\[TOC]

Classes and functions for building TensorFlow graphs.

## Core graph data structures

***

### `class tf.Graph` <a href="#graph" id="graph"></a>

A TensorFlow computation, represented as a dataflow graph.

A `Graph` contains a set of [`Operation`](framework.md#Operation) objects, which represent units of computation; and [`Tensor`](framework.md#Tensor) objects, which represent the units of data that flow between operations.

A default `Graph` is always registered, and accessible by calling [`tf.get_default_graph()`](framework.md#get\_default\_graph). To add an operation to the default graph, simply call one of the functions that defines a new `Operation`:

```
c = tf.constant(4.0)
assert c.graph is tf.get_default_graph()
```

Another typical usage involves the [`Graph.as_default()`](framework.md#Graph.as\_default) context manager, which overrides the current default graph for the lifetime of the context:

```python
g = tf.Graph()
with g.as_default():
  # Define operations and tensors in `g`.
  c = tf.constant(30.0)
  assert c.graph is g
```

Important note: This class _is not_ thread-safe for graph construction. All operations should be created from a single thread, or external synchronization must be provided. Unless otherwise specified, all methods are not thread-safe.

***

#### `tf.Graph.__init__()` {#Graph.**init**}

Creates a new, empty Graph.

***

#### `tf.Graph.as_default()` <a href="#graph.as_default" id="graph.as_default"></a>

Returns a context manager that makes this `Graph` the default graph.

This method should be used if you want to create multiple graphs in the same process. For convenience, a global default graph is provided, and all ops will be added to this graph if you do not create a new graph explicitly. Use this method with the `with` keyword to specify that ops created within the scope of a block should be added to this graph.

The default graph is a property of the current thread. If you create a new thread, and wish to use the default graph in that thread, you must explicitly add a `with g.as_default():` in that thread's function.

The following code examples are equivalent:

```python
# 1. Using Graph.as_default():
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g
```

**Returns:**

A context manager for using this graph as the default graph.

***

#### `tf.Graph.as_graph_def(from_version=None, add_shapes=False)` <a href="#graph.as_graph_def" id="graph.as_graph_def"></a>

Returns a serialized `GraphDef` representation of this graph.

The serialized `GraphDef` can be imported into another `Graph` (using [`import_graph_def()`](framework.md#import\_graph\_def)) or used with the [C++ Session API](broken-reference).

This method is thread-safe.

**Args:**

* `from_version`: Optional. If this is set, returns a `GraphDef` containing only the nodes that were added to this graph since its `version` property had the given value.
* `add_shapes`: If true, adds an "\_output\_shapes" list attr to each node with the inferred shapes of each of its outputs.

**Returns:**

A [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto) protocol buffer.

**Raises:**

* `ValueError`: If the `graph_def` would be too large.

***

#### `tf.Graph.finalize()` <a href="#graph.finalize" id="graph.finalize"></a>

Finalizes this graph, making it read-only.

After calling `g.finalize()`, no new operations can be added to `g`. This method is used to ensure that no operations are added to a graph when it is shared between multiple threads, for example when using a [`QueueRunner`](broken-reference).

***

#### `tf.Graph.finalized` <a href="#graph.finalized" id="graph.finalized"></a>

True if this graph has been finalized.

***

#### `tf.Graph.control_dependencies(control_inputs)` <a href="#graph.control_dependencies" id="graph.control_dependencies"></a>

Returns a context manager that specifies control dependencies.

Use with the `with` keyword to specify that all operations constructed within the context should have control dependencies on `control_inputs`. For example:

```python
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed.
  d = ...
  e = ...
```

Multiple calls to `control_dependencies()` can be nested, and in that case a new `Operation` will have control dependencies on the union of `control_inputs` from all active contexts.

```python
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies([c, d]):
    # Ops constructed here run after `a`, `b`, `c`, and `d`.
```

You can pass None to clear the control dependencies:

```python
with g.control_dependencies([a, b]):
  # Ops constructed here run after `a` and `b`.
  with g.control_dependencies(None):
    # Ops constructed here run normally, not waiting for either `a` or `b`.
    with g.control_dependencies([c, d]):
      # Ops constructed here run after `c` and `d`, also not waiting
      # for either `a` or `b`.
```

_N.B._ The control dependencies context applies _only_ to ops that are constructed within the context. Merely using an op or tensor in the context does not add a control dependency. The following example illustrates this point:

```python
# WRONG
def my_func(pred, tensor):
  t = tf.matmul(tensor, tensor)
  with tf.control_dependencies([pred]):
    # The matmul op is created outside the context, so no control
    # dependency will be added.
    return t

# RIGHT
def my_func(pred, tensor):
  with tf.control_dependencies([pred]):
    # The matmul op is created in the context, so a control dependency
    # will be added.
    return tf.matmul(tensor, tensor)
```

**Args:**

* `control_inputs`: A list of `Operation` or `Tensor` objects which must be executed or computed before running the operations defined in the context. Can also be `None` to clear the control dependencies.

**Returns:**

A context manager that specifies control dependencies for all operations constructed within the context.

**Raises:**

* `TypeError`: If `control_inputs` is not a list of `Operation` or `Tensor` objects.

***

#### `tf.Graph.device(device_name_or_function)` <a href="#graph.device" id="graph.device"></a>

Returns a context manager that specifies the default device to use.

The `device_name_or_function` argument may either be a device name string, a device function, or None:

* If it is a device name string, all operations constructed in this context will be assigned to the device with that name, unless overridden by a nested `device()` context.
* If it is a function, it will be treated as a function from Operation objects to device name strings, and invoked each time a new Operation is created. The Operation will be assigned to the device with the returned name.
* If it is None, all `device()` invocations from the enclosing context will be ignored.

For information about the valid syntax of device name strings, see the documentation in [`DeviceNameUtils`](https://www.tensorflow.org/code/tensorflow/core/util/device\_name\_utils.h).

For example:

```python
with g.device('/gpu:0'):
  # All operations constructed in this context will be placed
  # on GPU 0.
  with g.device(None):
    # All operations constructed in this context will have no
    # assigned device.

# Defines a function from `Operation` to device string.
def matmul_on_gpu(n):
  if n.type == "MatMul":
    return "/gpu:0"
  else:
    return "/cpu:0"

with g.device(matmul_on_gpu):
  # All operations of type "MatMul" constructed in this context
  # will be placed on GPU 0; all other operations will be placed
  # on CPU 0.
```

**N.B.** The device scope may be overridden by op wrappers or other library code. For example, a variable assignment op `v.assign()` must be colocated with the `tf.Variable` `v`, and incompatible device scopes will be ignored.

**Args:**

* `device_name_or_function`: The device name or function to use in the context.

**Returns:**

A context manager that specifies the default device to use for newly created ops.

***

#### `tf.Graph.name_scope(name)` <a href="#graph.name_scope" id="graph.name_scope"></a>

Returns a context manager that creates hierarchical names for operations.

A graph maintains a stack of name scopes. A `with name_scope(...):` statement pushes a new name onto the stack for the lifetime of the context.

The `name` argument will be interpreted as follows:

* A string (not ending with '/') will create a new name scope, in which `name` is appended to the prefix of all operations created in the context. If `name` has been used before, it will be made unique by calling `self.unique_name(name)`.
* A scope previously captured from a `with g.name_scope(...) as scope:` statement will be treated as an "absolute" name scope, which makes it possible to re-enter existing scopes.
* A value of `None` or the empty string will reset the current name scope to the top-level (empty) name scope.

For example:

```python
with tf.Graph().as_default() as g:
  c = tf.constant(5.0, name="c")
  assert c.op.name == "c"
  c_1 = tf.constant(6.0, name="c")
  assert c_1.op.name == "c_1"

  # Creates a scope called "nested"
  with g.name_scope("nested") as scope:
    nested_c = tf.constant(10.0, name="c")
    assert nested_c.op.name == "nested/c"

    # Creates a nested scope called "inner".
    with g.name_scope("inner"):
      nested_inner_c = tf.constant(20.0, name="c")
      assert nested_inner_c.op.name == "nested/inner/c"

    # Create a nested scope called "inner_1".
    with g.name_scope("inner"):
      nested_inner_1_c = tf.constant(30.0, name="c")
      assert nested_inner_1_c.op.name == "nested/inner_1/c"

      # Treats `scope` as an absolute name scope, and
      # switches to the "nested/" scope.
      with g.name_scope(scope):
        nested_d = tf.constant(40.0, name="d")
        assert nested_d.op.name == "nested/d"

        with g.name_scope(""):
          e = tf.constant(50.0, name="e")
          assert e.op.name == "e"
```

The name of the scope itself can be captured by `with g.name_scope(...) as scope:`, which stores the name of the scope in the variable `scope`. This value can be used to name an operation that represents the overall result of executing the ops in a scope. For example:

```python
inputs = tf.constant(...)
with g.name_scope('my_layer') as scope:
  weights = tf.Variable(..., name="weights")
  biases = tf.Variable(..., name="biases")
  affine = tf.matmul(inputs, weights) + biases
  output = tf.nn.relu(affine, name=scope)
```

**Args:**

* `name`: A name for the scope.

**Returns:**

A context manager that installs `name` as a new name scope.

A `Graph` instance supports an arbitrary number of "collections" that are identified by name. For convenience when building a large graph, collections can store groups of related objects: for example, the `tf.Variable` uses a collection (named [`tf.GraphKeys.VARIABLES`](framework.md#GraphKeys)) for all variables that are created during the construction of a graph. The caller may define additional collections by specifying a new name.

***

#### `tf.Graph.add_to_collection(name, value)` <a href="#graph.add_to_collection" id="graph.add_to_collection"></a>

Stores `value` in the collection with the given `name`.

Note that collections are not sets, so it is possible to add a value to a collection several times.

**Args:**

* `name`: The key for the collection. The `GraphKeys` class contains many standard names for collections.
* `value`: The value to add to the collection.

***

#### `tf.Graph.add_to_collections(names, value)` <a href="#graph.add_to_collections" id="graph.add_to_collections"></a>

Stores `value` in the collections given by `names`.

Note that collections are not sets, so it is possible to add a value to a collection several times. This function makes sure that duplicates in `names` are ignored, but it will not check for pre-existing membership of `value` in any of the collections in `names`.

`names` can be any iterable, but if `names` is a string, it is treated as a single collection name.

**Args:**

* `names`: The keys for the collections to add to. The `GraphKeys` class contains many standard names for collections.
* `value`: The value to add to the collections.

***

#### `tf.Graph.get_collection(name, scope=None)` <a href="#graph.get_collection" id="graph.get_collection"></a>

Returns a list of values in the collection with the given `name`.

This is different from `get_collection_ref()` which always returns the actual collection list if it exists in that it returns a new list each time it is called.

**Args:**

* `name`: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.
* `scope`: (Optional.) If supplied, the resulting list is filtered to include only items whose `name` attribute matches using `re.match`. Items without a `name` attribute are never returned if a scope is supplied and the choice or `re.match` means that a `scope` without special tokens filters by prefix.

**Returns:**

The list of values in the collection with the given `name`, or an empty list if no value has been added to that collection. The list contains the values in the order under which they were collected.

***

#### `tf.Graph.get_collection_ref(name)` <a href="#graph.get_collection_ref" id="graph.get_collection_ref"></a>

Returns a list of values in the collection with the given `name`.

If the collection exists, this returns the list itself, which can be modified in place to change the collection. If the collection does not exist, it is created as an empty list and the list is returned.

This is different from `get_collection()` which always returns a copy of the collection list if it exists and never creates an empty collection.

**Args:**

* `name`: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.

**Returns:**

The list of values in the collection with the given `name`, or an empty list if no value has been added to that collection.

***

#### `tf.Graph.as_graph_element(obj, allow_tensor=True, allow_operation=True)` <a href="#graph.as_graph_element" id="graph.as_graph_element"></a>

Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

This function validates that `obj` represents an element of this graph, and gives an informative error message if it is not.

This function is the canonical way to get/validate an object of one of the allowed types from an external argument reference in the Session API.

This method may be called concurrently from multiple threads.

**Args:**

* `obj`: A `Tensor`, an `Operation`, or the name of a tensor or operation. Can also be any object with an `_as_graph_element()` method that returns a value of one of these types.
* `allow_tensor`: If true, `obj` may refer to a `Tensor`.
* `allow_operation`: If true, `obj` may refer to an `Operation`.

**Returns:**

The `Tensor` or `Operation` in the Graph corresponding to `obj`.

**Raises:**

* `TypeError`: If `obj` is not a type we support attempting to convert to types.
* `ValueError`: If `obj` is of an appropriate type but invalid. For example, an invalid string.
* `KeyError`: If `obj` is not an object in the graph.

***

#### `tf.Graph.get_operation_by_name(name)` <a href="#graph.get_operation_by_name" id="graph.get_operation_by_name"></a>

Returns the `Operation` with the given `name`.

This method may be called concurrently from multiple threads.

**Args:**

* `name`: The name of the `Operation` to return.

**Returns:**

The `Operation` with the given `name`.

**Raises:**

* `TypeError`: If `name` is not a string.
* `KeyError`: If `name` does not correspond to an operation in this graph.

***

#### `tf.Graph.get_tensor_by_name(name)` <a href="#graph.get_tensor_by_name" id="graph.get_tensor_by_name"></a>

Returns the `Tensor` with the given `name`.

This method may be called concurrently from multiple threads.

**Args:**

* `name`: The name of the `Tensor` to return.

**Returns:**

The `Tensor` with the given `name`.

**Raises:**

* `TypeError`: If `name` is not a string.
* `KeyError`: If `name` does not correspond to a tensor in this graph.

***

#### `tf.Graph.get_operations()` <a href="#graph.get_operations" id="graph.get_operations"></a>

Return the list of operations in the graph.

You can modify the operations in place, but modifications to the list such as inserts/delete have no effect on the list of operations known to the graph.

This method may be called concurrently from multiple threads.

**Returns:**

A list of Operations.

***

#### `tf.Graph.seed` <a href="#graph.seed" id="graph.seed"></a>

The graph-level random seed of this graph.

***

#### `tf.Graph.unique_name(name, mark_as_used=True)` <a href="#graph.unique_name" id="graph.unique_name"></a>

Return a unique operation name for `name`.

Note: You rarely need to call `unique_name()` directly. Most of the time you just need to create `with g.name_scope()` blocks to generate structured names.

`unique_name` is used to generate structured names, separated by `"/"`, to help identify operations when debugging a graph. Operation names are displayed in error messages reported by the TensorFlow runtime, and in various visualization tools such as TensorBoard.

If `mark_as_used` is set to `True`, which is the default, a new unique name is created and marked as in use. If it's set to `False`, the unique name is returned without actually being marked as used. This is useful when the caller simply wants to know what the name to be created will be.

**Args:**

* `name`: The name for an operation.
* `mark_as_used`: Whether to mark this name as being used.

**Returns:**

A string to be passed to `create_op()` that will be used to name the operation being created.

***

#### `tf.Graph.version` <a href="#graph.version" id="graph.version"></a>

Returns a version number that increases as ops are added to the graph.

Note that this is unrelated to the [GraphDef version](framework.md#Graph.graph\_def\_version).

***

#### `tf.Graph.graph_def_versions` <a href="#graph.graph_def_versions" id="graph.graph_def_versions"></a>

The GraphDef version information of this graph.

For details on the meaning of each version, see \[`GraphDef`] (https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).

**Returns:**

A `VersionDef`.

***

#### `tf.Graph.create_op(op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True, compute_device=True)` <a href="#graph.create_op" id="graph.create_op"></a>

Creates an `Operation` in this graph.

This is a low-level interface for creating an `Operation`. Most programs will not call this method directly, and instead use the Python op constructors, such as `tf.constant()`, which add ops to the default graph.

**Args:**

* `op_type`: The `Operation` type to create. This corresponds to the `OpDef.name` field for the proto that defines the operation.
* `inputs`: A list of `Tensor` objects that will be inputs to the `Operation`.
* `dtypes`: A list of `DType` objects that will be the types of the tensors that the operation produces.
* `input_types`: (Optional.) A list of `DType`s that will be the types of the tensors that the operation consumes. By default, uses the base `DType` of each input in `inputs`. Operations that expect reference-typed inputs must specify `input_types` explicitly.
* `name`: (Optional.) A string name for the operation. If not specified, a name is generated based on `op_type`.
* `attrs`: (Optional.) A dictionary where the key is the attribute name (a string) and the value is the respective `attr` attribute of the `NodeDef` proto that will represent the operation (an `AttrValue` proto).
* `op_def`: (Optional.) The `OpDef` proto that describes the `op_type` that the operation will have.
* `compute_shapes`: (Optional.) If True, shape inference will be performed to compute the shapes of the outputs.
* `compute_device`: (Optional.) If True, device functions will be executed to compute the device property of the Operation.

**Raises:**

* `TypeError`: if any of the inputs is not a `Tensor`.
* `ValueError`: if colocation conflicts with existing device assignment.

**Returns:**

An `Operation` object.

***

#### `tf.Graph.gradient_override_map(op_type_map)` <a href="#graph.gradient_override_map" id="graph.gradient_override_map"></a>

EXPERIMENTAL: A context manager for overriding gradient functions.

This context manager can be used to override the gradient function that will be used for ops within the scope of the context.

For example:

```python
@tf.RegisterGradient("CustomSquare")
def _custom_square_grad(op, grad):
  # ...

with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  s_1 = tf.square(c)  # Uses the default gradient for tf.square.
  with g.gradient_override_map({"Square": "CustomSquare"}):
    s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                          # gradient of s_2.
```

**Args:**

* `op_type_map`: A dictionary mapping op type strings to alternative op type strings.

**Returns:**

A context manager that sets the alternative op type to be used for one or more ops created in that context.

**Raises:**

* `TypeError`: If `op_type_map` is not a dictionary mapping strings to strings.

#### Other Methods

***

#### `tf.Graph.colocate_with(op, ignore_existing=False)` <a href="#graph.colocate_with" id="graph.colocate_with"></a>

Returns a context manager that specifies an op to colocate with.

Note: this function is not for public use, only for internal libraries.

For example:

```python
a = tf.Variable([1.0])
with g.colocate_with(a):
  b = tf.constant(1.0)
  c = tf.add(a, b)
```

`b` and `c` will always be colocated with `a`, no matter where `a` is eventually placed.

**Args:**

* `op`: The op to colocate all created ops with.
* `ignore_existing`: If true, only applies colocation of this op within the context, rather than applying all colocation properties on the stack.

**Raises:**

* `ValueError`: if op is None.

**Yields:**

A context manager that specifies the op with which to colocate newly created ops.

***

#### `tf.Graph.get_all_collection_keys()` <a href="#graph.get_all_collection_keys" id="graph.get_all_collection_keys"></a>

Returns a list of collections used in this graph.

***

#### `tf.Graph.is_feedable(tensor)` <a href="#graph.is_feedable" id="graph.is_feedable"></a>

Returns `True` if and only if `tensor` is feedable.

***

#### `tf.Graph.is_fetchable(tensor_or_op)` <a href="#graph.is_fetchable" id="graph.is_fetchable"></a>

Returns `True` if and only if `tensor_or_op` is fetchable.

***

#### `tf.Graph.prevent_feeding(tensor)` <a href="#graph.prevent_feeding" id="graph.prevent_feeding"></a>

Marks the given `tensor` as unfeedable in this graph.

***

#### `tf.Graph.prevent_fetching(op)` <a href="#graph.prevent_fetching" id="graph.prevent_fetching"></a>

Marks the given `op` as unfetchable in this graph.

***

### `class tf.Operation` <a href="#operation" id="operation"></a>

Represents a graph node that performs computation on tensors.

An `Operation` is a node in a TensorFlow `Graph` that takes zero or more `Tensor` objects as input, and produces zero or more `Tensor` objects as output. Objects of type `Operation` are created by calling a Python op constructor (such as [`tf.matmul()`](broken-reference)) or [`Graph.create_op()`](framework.md#Graph.create\_op).

For example `c = tf.matmul(a, b)` creates an `Operation` of type "MatMul" that takes tensors `a` and `b` as input, and produces `c` as output.

After the graph has been launched in a session, an `Operation` can be executed by passing it to [`Session.run()`](broken-reference). `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

***

#### `tf.Operation.name` <a href="#operation.name" id="operation.name"></a>

The full name of this operation.

***

#### `tf.Operation.type` <a href="#operation.type" id="operation.type"></a>

The type of the op (e.g. `"MatMul"`).

***

#### `tf.Operation.inputs` <a href="#operation.inputs" id="operation.inputs"></a>

The list of `Tensor` objects representing the data inputs of this op.

***

#### `tf.Operation.control_inputs` <a href="#operation.control_inputs" id="operation.control_inputs"></a>

The `Operation` objects on which this op has a control dependency.

Before this op is executed, TensorFlow will ensure that the operations in `self.control_inputs` have finished executing. This mechanism can be used to run ops sequentially for performance reasons, or to ensure that the side effects of an op are observed in the correct order.

**Returns:**

A list of `Operation` objects.

***

#### `tf.Operation.outputs` <a href="#operation.outputs" id="operation.outputs"></a>

The list of `Tensor` objects representing the outputs of this op.

***

#### `tf.Operation.device` <a href="#operation.device" id="operation.device"></a>

The name of the device to which this op has been assigned, if any.

**Returns:**

The string name of the device to which this op has been assigned, or an empty string if it has not been assigned to a device.

***

#### `tf.Operation.graph` <a href="#operation.graph" id="operation.graph"></a>

The `Graph` that contains this operation.

***

#### `tf.Operation.run(feed_dict=None, session=None)` <a href="#operation.run" id="operation.run"></a>

Runs this operation in a `Session`.

Calling this method will execute all preceding operations that produce the inputs needed for this operation.

_N.B._ Before invoking `Operation.run()`, its graph must have been launched in a session, and either a default session must be available, or `session` must be specified explicitly.

**Args:**

* `feed_dict`: A dictionary that maps `Tensor` objects to feed values. See [`Session.run()`](broken-reference) for a description of the valid feed values.
* `session`: (Optional.) The `Session` to be used to run to this operation. If none, the default session will be used.

***

#### `tf.Operation.get_attr(name)` <a href="#operation.get_attr" id="operation.get_attr"></a>

Returns the value of the attr of this op with the given `name`.

**Args:**

* `name`: The name of the attr to fetch.

**Returns:**

The value of the attr, as a Python object.

**Raises:**

* `ValueError`: If this op does not have an attr with the given `name`.

***

#### `tf.Operation.traceback` <a href="#operation.traceback" id="operation.traceback"></a>

Returns the call stack from when this operation was constructed.

#### Other Methods

***

#### `tf.Operation.__init__(node_def, g, inputs=None, output_types=None, control_inputs=None, input_types=None, original_op=None, op_def=None)` {#Operation.**init**}

Creates an `Operation`.

NOTE: This constructor validates the name of the `Operation` (passed as `node_def.name`). Valid `Operation` names match the following regular expression:

```
[A-Za-z0-9.][A-Za-z0-9_.\-/]*
```

**Args:**

* `node_def`: `graph_pb2.NodeDef`. `NodeDef` for the `Operation`. Used for attributes of `graph_pb2.NodeDef`, typically `name`, `op`, and `device`. The `input` attribute is irrelevant here as it will be computed when generating the model.
* `g`: `Graph`. The parent graph.
* `inputs`: list of `Tensor` objects. The inputs to this `Operation`.
* `output_types`: list of `DType` objects. List of the types of the `Tensors` computed by this operation. The length of this list indicates the number of output endpoints of the `Operation`.
* `control_inputs`: list of operations or tensors from which to have a control dependency.
* `input_types`: List of `DType` objects representing the types of the tensors accepted by the `Operation`. By default uses `[x.dtype.base_dtype for x in inputs]`. Operations that expect reference-typed inputs must specify these explicitly.
* `original_op`: Optional. Used to associate the new `Operation` with an existing `Operation` (for example, a replica with the op that was replicated).
* `op_def`: Optional. The `op_def_pb2.OpDef` proto that describes the op type that this `Operation` represents.

**Raises:**

* `TypeError`: if control inputs are not Operations or Tensors, or if `node_def` is not a `NodeDef`, or if `g` is not a `Graph`, or if `inputs` are not tensors, or if `inputs` and `input_types` are incompatible.
* `ValueError`: if the `node_def` name is not valid.

***

#### `tf.Operation.colocation_groups()` <a href="#operation.colocation_groups" id="operation.colocation_groups"></a>

Returns the list of colocation groups of the op.

***

#### `tf.Operation.node_def` <a href="#operation.node_def" id="operation.node_def"></a>

Returns a serialized `NodeDef` representation of this operation.

**Returns:**

A [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto) protocol buffer.

***

#### `tf.Operation.op_def` <a href="#operation.op_def" id="operation.op_def"></a>

Returns the `OpDef` proto that represents the type of this op.

**Returns:**

An [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op\_def.proto) protocol buffer.

***

#### `tf.Operation.values()` <a href="#operation.values" id="operation.values"></a>

DEPRECATED: Use outputs.

***

### `class tf.Tensor` <a href="#tensor" id="tensor"></a>

Represents a value produced by an `Operation`.

A `Tensor` is a symbolic handle to one of the outputs of an `Operation`. It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow [`Session`](broken-reference).

This class has two primary purposes:

1. A `Tensor` can be passed as an input to another `Operation`. This builds a dataflow connection between operations, which enables TensorFlow to execute an entire `Graph` that represents a large, multi-step computation.
2. After the graph has been launched in a session, the value of the `Tensor` can be computed by passing it to [`Session.run()`](broken-reference). `t.eval()` is a shortcut for calling `tf.get_default_session().run(t)`.

In the following example, `c`, `d`, and `e` are symbolic `Tensor` objects, whereas `result` is a numpy array that stores a concrete value:

```python
# Build a dataflow graph.
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph and store the value that `e` represents in `result`.
result = sess.run(e)
```

***

#### `tf.Tensor.dtype` <a href="#tensor.dtype" id="tensor.dtype"></a>

The `DType` of elements in this tensor.

***

#### `tf.Tensor.name` <a href="#tensor.name" id="tensor.name"></a>

The string name of this tensor.

***

#### `tf.Tensor.value_index` <a href="#tensor.value_index" id="tensor.value_index"></a>

The index of this tensor in the outputs of its `Operation`.

***

#### `tf.Tensor.graph` <a href="#tensor.graph" id="tensor.graph"></a>

The `Graph` that contains this tensor.

***

#### `tf.Tensor.op` <a href="#tensor.op" id="tensor.op"></a>

The `Operation` that produces this tensor as an output.

***

#### `tf.Tensor.consumers()` <a href="#tensor.consumers" id="tensor.consumers"></a>

Returns a list of `Operation`s that consume this tensor.

**Returns:**

A list of `Operation`s.

***

#### `tf.Tensor.eval(feed_dict=None, session=None)` <a href="#tensor.eval" id="tensor.eval"></a>

Evaluates this tensor in a `Session`.

Calling this method will execute all preceding operations that produce the inputs needed for the operation that produces this tensor.

_N.B._ Before invoking `Tensor.eval()`, its graph must have been launched in a session, and either a default session must be available, or `session` must be specified explicitly.

**Args:**

* `feed_dict`: A dictionary that maps `Tensor` objects to feed values. See [`Session.run()`](broken-reference) for a description of the valid feed values.
* `session`: (Optional.) The `Session` to be used to evaluate this tensor. If none, the default session will be used.

**Returns:**

A numpy array corresponding to the value of this tensor.

***

#### `tf.Tensor.get_shape()` <a href="#tensor.get_shape" id="tensor.get_shape"></a>

Returns the `TensorShape` that represents the shape of this tensor.

The shape is computed using shape inference functions that are registered for each `Operation` type using `tf.RegisterShape`. See [`TensorShape`](framework.md#TensorShape) for more details of what a shape represents.

The inferred shape of a tensor is used to provide shape information without having to launch the graph in a session. This can be used for debugging, and providing early error messages. For example:

```python
c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(c.get_shape())
==> TensorShape([Dimension(2), Dimension(3)])

d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

print(d.get_shape())
==> TensorShape([Dimension(4), Dimension(2)])

# Raises a ValueError, because `c` and `d` do not have compatible
# inner dimensions.
e = tf.matmul(c, d)

f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

print(f.get_shape())
==> TensorShape([Dimension(3), Dimension(4)])
```

In some cases, the inferred shape may have unknown dimensions. If the caller has additional information about the values of these dimensions, `Tensor.set_shape()` can be used to augment the inferred shape.

**Returns:**

A `TensorShape` representing the shape of this tensor.

***

#### `tf.Tensor.set_shape(shape)` <a href="#tensor.set_shape" id="tensor.set_shape"></a>

Updates the shape of this tensor.

This method can be called multiple times, and will merge the given `shape` with the current shape of this tensor. It can be used to provide additional information about the shape of this tensor that cannot be inferred from the graph alone. For example, this can be used to provide additional information about the shapes of images:

```python
_, image_data = tf.TFRecordReader(...).read(...)
image = tf.image.decode_png(image_data, channels=3)

# The height and width dimensions of `image` are data dependent, and
# cannot be computed without executing the op.
print(image.get_shape())
==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

# We know that each image in this dataset is 28 x 28 pixels.
image.set_shape([28, 28, 3])
print(image.get_shape())
==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
```

**Args:**

* `shape`: A `TensorShape` representing the shape of this tensor.

**Raises:**

* `ValueError`: If `shape` is not compatible with the current shape of this tensor.

#### Other Methods

***

#### `tf.Tensor.__init__(op, value_index, dtype)` {#Tensor.**init**}

Creates a new `Tensor`.

**Args:**

* `op`: An `Operation`. `Operation` that computes this tensor.
* `value_index`: An `int`. Index of the operation's endpoint that produces this tensor.
* `dtype`: A `DType`. Type of elements stored in this tensor.

**Raises:**

* `TypeError`: If the op is not an `Operation`.

***

#### `tf.Tensor.device` <a href="#tensor.device" id="tensor.device"></a>

The name of the device on which this tensor will be produced, or None.

## Tensor types

***

### `class tf.DType` <a href="#dtype" id="dtype"></a>

Represents the type of the elements in a `Tensor`.

The following `DType` objects are defined:

* `tf.float16`: 16-bit half-precision floating-point.
* `tf.float32`: 32-bit single-precision floating-point.
* `tf.float64`: 64-bit double-precision floating-point.
* `tf.bfloat16`: 16-bit truncated floating-point.
* `tf.complex64`: 64-bit single-precision complex.
* `tf.complex128`: 128-bit double-precision complex.
* `tf.int8`: 8-bit signed integer.
* `tf.uint8`: 8-bit unsigned integer.
* `tf.uint16`: 16-bit unsigned integer.
* `tf.int16`: 16-bit signed integer.
* `tf.int32`: 32-bit signed integer.
* `tf.int64`: 64-bit signed integer.
* `tf.bool`: Boolean.
* `tf.string`: String.
* `tf.qint8`: Quantized 8-bit signed integer.
* `tf.quint8`: Quantized 8-bit unsigned integer.
* `tf.qint16`: Quantized 16-bit signed integer.
* `tf.quint16`: Quantized 16-bit unsigned integer.
* `tf.qint32`: Quantized 32-bit signed integer.

In addition, variants of these types with the `_ref` suffix are defined for reference-typed tensors.

The `tf.as_dtype()` function converts numpy types and string type names to a `DType` object.

***

#### `tf.DType.is_compatible_with(other)` <a href="#dtype.is_compatible_with" id="dtype.is_compatible_with"></a>

Returns True if the `other` DType will be converted to this DType.

The conversion rules are as follows:

```
DType(T)       .is_compatible_with(DType(T))        == True
DType(T)       .is_compatible_with(DType(T).as_ref) == True
DType(T).as_ref.is_compatible_with(DType(T))        == False
DType(T).as_ref.is_compatible_with(DType(T).as_ref) == True
```

**Args:**

* `other`: A `DType` (or object that may be converted to a `DType`).

**Returns:**

True if a Tensor of the `other` `DType` will be implicitly converted to this `DType`.

***

#### `tf.DType.name` <a href="#dtype.name" id="dtype.name"></a>

Returns the string name for this `DType`.

***

#### `tf.DType.base_dtype` <a href="#dtype.base_dtype" id="dtype.base_dtype"></a>

Returns a non-reference `DType` based on this `DType`.

***

#### `tf.DType.real_dtype` <a href="#dtype.real_dtype" id="dtype.real_dtype"></a>

Returns the dtype correspond to this dtype's real part.

***

#### `tf.DType.is_ref_dtype` <a href="#dtype.is_ref_dtype" id="dtype.is_ref_dtype"></a>

Returns `True` if this `DType` represents a reference type.

***

#### `tf.DType.as_ref` <a href="#dtype.as_ref" id="dtype.as_ref"></a>

Returns a reference `DType` based on this `DType`.

***

#### `tf.DType.is_floating` <a href="#dtype.is_floating" id="dtype.is_floating"></a>

Returns whether this is a (real) floating point type.

***

#### `tf.DType.is_complex` <a href="#dtype.is_complex" id="dtype.is_complex"></a>

Returns whether this is a complex floating point type.

***

#### `tf.DType.is_integer` <a href="#dtype.is_integer" id="dtype.is_integer"></a>

Returns whether this is a (non-quantized) integer type.

***

#### `tf.DType.is_quantized` <a href="#dtype.is_quantized" id="dtype.is_quantized"></a>

Returns whether this is a quantized data type.

***

#### `tf.DType.is_unsigned` <a href="#dtype.is_unsigned" id="dtype.is_unsigned"></a>

Returns whether this type is unsigned.

Non-numeric, unordered, and quantized types are not considered unsigned, and this function returns `False`.

**Returns:**

Whether a `DType` is unsigned.

***

#### `tf.DType.as_numpy_dtype` <a href="#dtype.as_numpy_dtype" id="dtype.as_numpy_dtype"></a>

Returns a `numpy.dtype` based on this `DType`.

***

#### `tf.DType.as_datatype_enum` <a href="#dtype.as_datatype_enum" id="dtype.as_datatype_enum"></a>

Returns a `types_pb2.DataType` enum value based on this `DType`.

#### Other Methods

***

#### `tf.DType.__init__(type_enum)` {#DType.**init**}

Creates a new `DataType`.

NOTE(mrry): In normal circumstances, you should not need to construct a `DataType` object directly. Instead, use the `tf.as_dtype()` function.

**Args:**

* `type_enum`: A `types_pb2.DataType` enum value.

**Raises:**

* `TypeError`: If `type_enum` is not a value `types_pb2.DataType`.

***

#### `tf.DType.max` <a href="#dtype.max" id="dtype.max"></a>

Returns the maximum representable value in this data type.

**Raises:**

* `TypeError`: if this is a non-numeric, unordered, or quantized type.

***

#### `tf.DType.min` <a href="#dtype.min" id="dtype.min"></a>

Returns the minimum representable value in this data type.

**Raises:**

* `TypeError`: if this is a non-numeric, unordered, or quantized type.

***

#### `tf.DType.size` <a href="#dtype.size" id="dtype.size"></a>

***

### `tf.as_dtype(type_value)` <a href="#as_dtype" id="as_dtype"></a>

Converts the given `type_value` to a `DType`.

**Args:**

* `type_value`: A value that can be converted to a `tf.DType` object. This may currently be a `tf.DType` object, a [`DataType` enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto), a string type name, or a `numpy.dtype`.

**Returns:**

A `DType` corresponding to `type_value`.

**Raises:**

* `TypeError`: If `type_value` cannot be converted to a `DType`.

## Utility functions

***

### `tf.device(device_name_or_function)` <a href="#device" id="device"></a>

Wrapper for `Graph.device()` using the default graph.

See [`Graph.device()`](framework.md#Graph.device) for more details.

**Args:**

* `device_name_or_function`: The device name or function to use in the context.

**Returns:**

A context manager that specifies the default device to use for newly created ops.

***

### `tf.name_scope(name)` <a href="#name_scope" id="name_scope"></a>

Wrapper for `Graph.name_scope()` using the default graph.

See [`Graph.name_scope()`](framework.md#Graph.name\_scope) for more details.

**Args:**

* `name`: A name for the scope.

**Returns:**

A context manager that installs `name` as a new name scope in the default graph.

***

### `tf.control_dependencies(control_inputs)` <a href="#control_dependencies" id="control_dependencies"></a>

Wrapper for `Graph.control_dependencies()` using the default graph.

See [`Graph.control_dependencies()`](framework.md#Graph.control\_dependencies) for more details.

**Args:**

* `control_inputs`: A list of `Operation` or `Tensor` objects which must be executed or computed before running the operations defined in the context. Can also be `None` to clear the control dependencies.

**Returns:**

A context manager that specifies control dependencies for all operations constructed within the context.

***

### `tf.convert_to_tensor(value, dtype=None, name=None, as_ref=False)` <a href="#convert_to_tensor" id="convert_to_tensor"></a>

Converts the given `value` to a `Tensor`.

This function converts Python objects of various types to `Tensor` objects. It accepts `Tensor` objects, numpy arrays, Python lists, and Python scalars. For example:

```python
import numpy as np

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

# The following calls are equivalent.
value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
```

This function can be useful when composing a new operation in Python (such as `my_func` in the example above). All standard Python op constructors apply this function to each of their Tensor-valued inputs, which allows those ops to accept numpy arrays, Python lists, and scalars in addition to `Tensor` objects.

**Args:**

* `value`: An object whose type has a registered `Tensor` conversion function.
* `dtype`: Optional element type for the returned tensor. If missing, the type is inferred from the type of `value`.
* `name`: Optional name to use if a new `Tensor` is created.
* `as_ref`: True if we want the result as a ref tensor. Only used if a new `Tensor` is created.

**Returns:**

A `Tensor` based on `value`.

**Raises:**

* `TypeError`: If no conversion function is registered for `value`.
* `RuntimeError`: If a registered conversion function returns an invalid value.

***

### `tf.convert_to_tensor_or_indexed_slices(value, dtype=None, name=None, as_ref=False)` <a href="#convert_to_tensor_or_indexed_slices" id="convert_to_tensor_or_indexed_slices"></a>

Converts the given object to a `Tensor` or an `IndexedSlices`.

If `value` is an `IndexedSlices` or `SparseTensor` it is returned unmodified. Otherwise, it is converted to a `Tensor` using `convert_to_tensor()`.

**Args:**

* `value`: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed by `convert_to_tensor()`.
* `dtype`: (Optional.) The required `DType` of the returned `Tensor` or `IndexedSlices`.
* `name`: (Optional.) A name to use if a new `Tensor` is created.
* `as_ref`: True if the caller wants the results as ref tensors.

**Returns:**

An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

**Raises:**

* `ValueError`: If `dtype` does not match the element type of `value`.

***

### `tf.get_default_graph()` <a href="#get_default_graph" id="get_default_graph"></a>

Returns the default graph for the current thread.

The returned graph will be the innermost graph on which a `Graph.as_default()` context has been entered, or a global default graph if none has been explicitly created.

NOTE: The default graph is a property of the current thread. If you create a new thread, and wish to use the default graph in that thread, you must explicitly add a `with g.as_default():` in that thread's function.

**Returns:**

The default `Graph` being used in the current thread.

***

### `tf.reset_default_graph()` <a href="#reset_default_graph" id="reset_default_graph"></a>

Clears the default graph stack and resets the global default graph.

NOTE: The default graph is a property of the current thread. This function applies only to the current thread. Calling this function while a `tf.Session` or `tf.InteractiveSession` is active will result in undefined behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects after calling this function will result in undefined behavior.

***

### `tf.import_graph_def(graph_def, input_map=None, return_elements=None, name=None, op_dict=None, producer_op_list=None)` <a href="#import_graph_def" id="import_graph_def"></a>

Imports the TensorFlow graph in `graph_def` into the Python `Graph`.

This function provides a way to import a serialized TensorFlow [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto) protocol buffer, and extract individual objects in the `GraphDef` as [`Tensor`](framework.md#Tensor) and [`Operation`](framework.md#Operation) objects. See [`Graph.as_graph_def()`](framework.md#Graph.as\_graph\_def) for a way to create a `GraphDef` proto.

**Args:**

* `graph_def`: A `GraphDef` proto containing operations to be imported into the default graph.
* `input_map`: A dictionary mapping input names (as strings) in `graph_def` to `Tensor` objects. The values of the named input tensors in the imported graph will be re-mapped to the respective `Tensor` values.
* `return_elements`: A list of strings containing operation names in `graph_def` that will be returned as `Operation` objects; and/or tensor names in `graph_def` that will be returned as `Tensor` objects.
* `name`: (Optional.) A prefix that will be prepended to the names in `graph_def`. Defaults to `"import"`.
* `op_dict`: (Optional.) A dictionary mapping op type names to `OpDef` protos. Must contain an `OpDef` proto for each op type named in `graph_def`. If omitted, uses the `OpDef` protos registered in the global registry.
* `producer_op_list`: (Optional.) An `OpList` proto with the (possibly stripped) list of `OpDef`s used by the producer of the graph. If provided, attrs for ops in `graph_def` that are not in `op_dict` that have their default value according to `producer_op_list` will be removed. This will allow some more `GraphDef`s produced by later binaries to be accepted by earlier binaries.

**Returns:**

A list of `Operation` and/or `Tensor` objects from the imported graph, corresponding to the names in `return_elements`.

**Raises:**

* `TypeError`: If `graph_def` is not a `GraphDef` proto, `input_map` is not a dictionary mapping strings to `Tensor` objects, or `return_elements` is not a list of strings.
* `ValueError`: If `input_map`, or `return_elements` contains names that do not appear in `graph_def`, or `graph_def` is not well-formed (e.g. it refers to an unknown tensor).

***

### `tf.load_file_system_library(library_filename)` <a href="#load_file_system_library" id="load_file_system_library"></a>

Loads a TensorFlow plugin, containing file system implementation.

Pass `library_filename` to a platform-specific mechanism for dynamically loading a library. The rules for determining the exact location of the library are platform-specific and are not documented here.

**Args:**

* `library_filename`: Path to the plugin. Relative or absolute filesystem path to a dynamic library file.

**Returns:**

None.

**Raises:**

* `RuntimeError`: when unable to load the library.

***

### `tf.load_op_library(library_filename)` <a href="#load_op_library" id="load_op_library"></a>

Loads a TensorFlow plugin, containing custom ops and kernels.

Pass "library\_filename" to a platform-specific mechanism for dynamically loading a library. The rules for determining the exact location of the library are platform-specific and are not documented here.

**Args:**

* `library_filename`: Path to the plugin. Relative or absolute filesystem path to a dynamic library file.

**Returns:**

A python module containing the Python wrappers for Ops defined in the plugin.

**Raises:**

* `RuntimeError`: when unable to load the library or get the python wrappers.

## Graph collections

***

### `tf.add_to_collection(name, value)` <a href="#add_to_collection" id="add_to_collection"></a>

Wrapper for `Graph.add_to_collection()` using the default graph.

See [`Graph.add_to_collection()`](framework.md#Graph.add\_to\_collection) for more details.

**Args:**

* `name`: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.
* `value`: The value to add to the collection.

***

### `tf.get_collection(key, scope=None)` <a href="#get_collection" id="get_collection"></a>

Wrapper for `Graph.get_collection()` using the default graph.

See [`Graph.get_collection()`](framework.md#Graph.get\_collection) for more details.

**Args:**

* `key`: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.
* `scope`: (Optional.) If supplied, the resulting list is filtered to include only items whose `name` attribute matches using `re.match`. Items without a `name` attribute are never returned if a scope is supplied and the choice or `re.match` means that a `scope` without special tokens filters by prefix.

**Returns:**

The list of values in the collection with the given `name`, or an empty list if no value has been added to that collection. The list contains the values in the order under which they were collected.

***

### `tf.get_collection_ref(key)` <a href="#get_collection_ref" id="get_collection_ref"></a>

Wrapper for `Graph.get_collection_ref()` using the default graph.

See [`Graph.get_collection_ref()`](framework.md#Graph.get\_collection\_ref) for more details.

**Args:**

* `key`: The key for the collection. For example, the `GraphKeys` class contains many standard names for collections.

**Returns:**

The list of values in the collection with the given `name`, or an empty list if no value has been added to that collection. Note that this returns the collection list itself, which can be modified in place to change the collection.

***

### `class tf.GraphKeys` <a href="#graphkeys" id="graphkeys"></a>

Standard names to use for graph collections.

The standard library uses various well-known names to collect and retrieve values associated with a graph. For example, the `tf.Optimizer` subclasses default to optimizing the variables collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is specified, but it is also possible to pass an explicit list of variables.

The following standard keys are defined:

* `VARIABLES`: the `Variable` objects that comprise a model, and must be saved and restored together. See [`tf.all_variables()`](broken-reference) for more details.
* `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will be trained by an optimizer. See [`tf.trainable_variables()`](broken-reference) for more details.
* `SUMMARIES`: the summary `Tensor` objects that have been created in the graph. See [`tf.merge_all_summaries()`](broken-reference) for more details.
* `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to produce input for a computation. See [`tf.start_queue_runners()`](broken-reference) for more details.
* `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also keep moving averages. See [`tf.moving_average_variables()`](broken-reference) for more details.
* `REGULARIZATION_LOSSES`: regularization losses collected during graph construction.
* `WEIGHTS`: weights inside neural network layers
* `BIASES`: biases inside neural network layers
* `ACTIVATIONS`: activations of neural network layers

## Defining new operations

***

### `class tf.RegisterGradient` <a href="#registergradient" id="registergradient"></a>

A decorator for registering the gradient function for an op type.

This decorator is only used when defining a new op type. For an op with `m` inputs and `n` outputs, the gradient function is a function that takes the original `Operation` and `n` `Tensor` objects (representing the gradients with respect to each output of the op), and returns `m` `Tensor` objects (representing the partial gradients with respect to each input of the op).

For example, assuming that operations of type `"Sub"` take two inputs `x` and `y`, and return a single output `x - y`, the following gradient function would be registered:

```python
@tf.RegisterGradient("Sub")
def _sub_grad(unused_op, grad):
  return grad, tf.neg(grad)
```

The decorator argument `op_type` is the string type of an operation. This corresponds to the `OpDef.name` field for the proto that defines the operation.

***

#### `tf.RegisterGradient.__init__(op_type)` {#RegisterGradient.**init**}

Creates a new decorator with `op_type` as the Operation type.

**Args:**

* `op_type`: The string type of an operation. This corresponds to the `OpDef.name` field for the proto that defines the operation.

***

### `tf.NoGradient(op_type)` <a href="#nogradient" id="nogradient"></a>

Specifies that ops of type `op_type` do not have a defined gradient.

This function is only used when defining a new op type. It may be used for ops such as `tf.size()` that are not differentiable. For example:

```python
tf.NoGradient("Size")
```

**Args:**

* `op_type`: The string type of an operation. This corresponds to the `OpDef.name` field for the proto that defines the operation.

**Raises:**

* `TypeError`: If `op_type` is not a string.

***

### `class tf.RegisterShape` <a href="#registershape" id="registershape"></a>

A decorator for registering the shape function for an op type.

This decorator is only used when defining a new op type. A shape function is a function from an `Operation` object to a list of `TensorShape` objects, with one `TensorShape` for each output of the operation.

For example, assuming that operations of type `"Sub"` take two inputs `x` and `y`, and return a single output `x - y`, all with the same shape, the following shape function would be registered:

```python
@tf.RegisterShape("Sub")
def _sub_shape(op):
  return [op.inputs[0].get_shape().merge_with(op.inputs[1].get_shape())]
```

The decorator argument `op_type` is the string type of an operation. This corresponds to the `OpDef.name` field for the proto that defines the operation.

***

#### `tf.RegisterShape.__init__(op_type)` {#RegisterShape.**init**}

Saves the `op_type` as the `Operation` type.

***

### `class tf.TensorShape` <a href="#tensorshape" id="tensorshape"></a>

Represents the shape of a `Tensor`.

A `TensorShape` represents a possibly-partial shape specification for a `Tensor`. It may be one of the following:

* _Fully-known shape:_ has a known number of dimensions and a known size for each dimension.
* _Partially-known shape:_ has a known number of dimensions, and an unknown size for one or more dimension.
* _Unknown shape:_ has an unknown number of dimensions, and an unknown size in all dimensions.

If a tensor is produced by an operation of type `"Foo"`, its shape may be inferred if there is a registered shape function for `"Foo"`. See [`tf.RegisterShape()`](framework.md#RegisterShape) for details of shape functions and how to register them. Alternatively, the shape may be set explicitly using [`Tensor.set_shape()`](framework.md#Tensor.set\_shape).

***

#### `tf.TensorShape.merge_with(other)` <a href="#tensorshape.merge_with" id="tensorshape.merge_with"></a>

Returns a `TensorShape` combining the information in `self` and `other`.

The dimensions in `self` and `other` are merged elementwise, according to the rules defined for `Dimension.merge_with()`.

**Args:**

* `other`: Another `TensorShape`.

**Returns:**

A `TensorShape` containing the combined information of `self` and `other`.

**Raises:**

* `ValueError`: If `self` and `other` are not compatible.

***

#### `tf.TensorShape.concatenate(other)` <a href="#tensorshape.concatenate" id="tensorshape.concatenate"></a>

Returns the concatenation of the dimension in `self` and `other`.

_N.B._ If either `self` or `other` is completely unknown, concatenation will discard information about the other shape. In future, we might support concatenation that preserves this information for use with slicing.

**Args:**

* `other`: Another `TensorShape`.

**Returns:**

A `TensorShape` whose dimensions are the concatenation of the dimensions in `self` and `other`.

***

#### `tf.TensorShape.ndims` <a href="#tensorshape.ndims" id="tensorshape.ndims"></a>

Returns the rank of this shape, or None if it is unspecified.

***

#### `tf.TensorShape.dims` <a href="#tensorshape.dims" id="tensorshape.dims"></a>

Returns a list of Dimensions, or None if the shape is unspecified.

***

#### `tf.TensorShape.as_list()` <a href="#tensorshape.as_list" id="tensorshape.as_list"></a>

Returns a list of integers or None for each dimension.

**Returns:**

A list of integers or None for each dimension.

***

#### `tf.TensorShape.as_proto()` <a href="#tensorshape.as_proto" id="tensorshape.as_proto"></a>

Returns this shape as a `TensorShapeProto`.

***

#### `tf.TensorShape.is_compatible_with(other)` <a href="#tensorshape.is_compatible_with" id="tensorshape.is_compatible_with"></a>

Returns True iff `self` is compatible with `other`.

Two possibly-partially-defined shapes are compatible if there exists a fully-defined shape that both shapes can represent. Thus, compatibility allows the shape inference code to reason about partially-defined shapes. For example:

* TensorShape(None) is compatible with all shapes.
* TensorShape(\[None, None]) is compatible with all two-dimensional shapes, such as TensorShape(\[32, 784]), and also TensorShape(None). It is not compatible with, for example, TensorShape(\[None]) or TensorShape(\[None, None, None]).
* TensorShape(\[32, None]) is compatible with all two-dimensional shapes with size 32 in the 0th dimension, and also TensorShape(\[None, None]) and TensorShape(None). It is not compatible with, for example, TensorShape(\[32]), TensorShape(\[32, None, 1]) or TensorShape(\[64, None]).
* TensorShape(\[32, 784]) is compatible with itself, and also TensorShape(\[32, None]), TensorShape(\[None, 784]), TensorShape(\[None, None]) and TensorShape(None). It is not compatible with, for example, TensorShape(\[32, 1, 784]) or TensorShape(\[None]).

The compatibility relation is reflexive and symmetric, but not transitive. For example, TensorShape(\[32, 784]) is compatible with TensorShape(None), and TensorShape(None) is compatible with TensorShape(\[4, 4]), but TensorShape(\[32, 784]) is not compatible with TensorShape(\[4, 4]).

**Args:**

* `other`: Another TensorShape.

**Returns:**

True iff `self` is compatible with `other`.

***

#### `tf.TensorShape.is_fully_defined()` <a href="#tensorshape.is_fully_defined" id="tensorshape.is_fully_defined"></a>

Returns True iff `self` is fully defined in every dimension.

***

#### `tf.TensorShape.with_rank(rank)` <a href="#tensorshape.with_rank" id="tensorshape.with_rank"></a>

Returns a shape based on `self` with the given rank.

This method promotes a completely unknown shape to one with a known rank.

**Args:**

* `rank`: An integer.

**Returns:**

A shape that is at least as specific as `self` with the given rank.

**Raises:**

* `ValueError`: If `self` does not represent a shape with the given `rank`.

***

#### `tf.TensorShape.with_rank_at_least(rank)` <a href="#tensorshape.with_rank_at_least" id="tensorshape.with_rank_at_least"></a>

Returns a shape based on `self` with at least the given rank.

**Args:**

* `rank`: An integer.

**Returns:**

A shape that is at least as specific as `self` with at least the given rank.

**Raises:**

* `ValueError`: If `self` does not represent a shape with at least the given `rank`.

***

#### `tf.TensorShape.with_rank_at_most(rank)` <a href="#tensorshape.with_rank_at_most" id="tensorshape.with_rank_at_most"></a>

Returns a shape based on `self` with at most the given rank.

**Args:**

* `rank`: An integer.

**Returns:**

A shape that is at least as specific as `self` with at most the given rank.

**Raises:**

* `ValueError`: If `self` does not represent a shape with at most the given `rank`.

***

#### `tf.TensorShape.assert_has_rank(rank)` <a href="#tensorshape.assert_has_rank" id="tensorshape.assert_has_rank"></a>

Raises an exception if `self` is not compatible with the given `rank`.

**Args:**

* `rank`: An integer.

**Raises:**

* `ValueError`: If `self` does not represent a shape with the given `rank`.

***

#### `tf.TensorShape.assert_same_rank(other)` <a href="#tensorshape.assert_same_rank" id="tensorshape.assert_same_rank"></a>

Raises an exception if `self` and `other` do not have compatible ranks.

**Args:**

* `other`: Another `TensorShape`.

**Raises:**

* `ValueError`: If `self` and `other` do not represent shapes with the same rank.

***

#### `tf.TensorShape.assert_is_compatible_with(other)` <a href="#tensorshape.assert_is_compatible_with" id="tensorshape.assert_is_compatible_with"></a>

Raises exception if `self` and `other` do not represent the same shape.

This method can be used to assert that there exists a shape that both `self` and `other` represent.

**Args:**

* `other`: Another TensorShape.

**Raises:**

* `ValueError`: If `self` and `other` do not represent the same shape.

***

#### `tf.TensorShape.assert_is_fully_defined()` <a href="#tensorshape.assert_is_fully_defined" id="tensorshape.assert_is_fully_defined"></a>

Raises an exception if `self` is not fully defined in every dimension.

**Raises:**

* `ValueError`: If `self` does not have a known value for every dimension.

#### Other Methods

***

#### `tf.TensorShape.__init__(dims)` {#TensorShape.**init**}

Creates a new TensorShape with the given dimensions.

**Args:**

* `dims`: A list of Dimensions, or None if the shape is unspecified.
* `DEPRECATED`: A single integer is treated as a singleton list.

**Raises:**

* `TypeError`: If dims cannot be converted to a list of dimensions.

***

#### `tf.TensorShape.num_elements()` <a href="#tensorshape.num_elements" id="tensorshape.num_elements"></a>

Returns the total number of elements, or none for incomplete shapes.

***

### `class tf.Dimension` <a href="#dimension" id="dimension"></a>

Represents the value of one dimension in a TensorShape.

***

#### `tf.Dimension.__init__(value)` {#Dimension.**init**}

Creates a new Dimension with the given value.

***

#### `tf.Dimension.assert_is_compatible_with(other)` <a href="#dimension.assert_is_compatible_with" id="dimension.assert_is_compatible_with"></a>

Raises an exception if `other` is not compatible with this Dimension.

**Args:**

* `other`: Another Dimension.

**Raises:**

* `ValueError`: If `self` and `other` are not compatible (see is\_compatible\_with).

***

#### `tf.Dimension.is_compatible_with(other)` <a href="#dimension.is_compatible_with" id="dimension.is_compatible_with"></a>

Returns true if `other` is compatible with this Dimension.

Two known Dimensions are compatible if they have the same value. An unknown Dimension is compatible with all other Dimensions.

**Args:**

* `other`: Another Dimension.

**Returns:**

True if this Dimension and `other` are compatible.

***

#### `tf.Dimension.merge_with(other)` <a href="#dimension.merge_with" id="dimension.merge_with"></a>

Returns a Dimension that combines the information in `self` and `other`.

Dimensions are combined as follows:

```
Dimension(n)   .merge_with(Dimension(n))    == Dimension(n)
Dimension(n)   .merge_with(Dimension(None)) == Dimension(n)
Dimension(None).merge_with(Dimension(n))    == Dimension(n)
Dimension(None).merge_with(Dimension(None)) == Dimension(None)
Dimension(n)   .merge_with(Dimension(m)) raises ValueError for n != m
```

**Args:**

* `other`: Another Dimension.

**Returns:**

A Dimension containing the combined information of `self` and `other`.

**Raises:**

* `ValueError`: If `self` and `other` are not compatible (see is\_compatible\_with).

***

#### `tf.Dimension.value` <a href="#dimension.value" id="dimension.value"></a>

The value of this dimension, or None if it is unknown.

***

### `tf.op_scope(values, name, default_name=None)` <a href="#op_scope" id="op_scope"></a>

Returns a context manager for use when defining a Python op.

This context manager validates that the given `values` are from the same graph, ensures that graph is the default graph, and pushes a name scope.

For example, to define a new Python op called `my_op`:

```python
def my_op(a, b, c, name=None):
  with tf.op_scope([a, b, c], name, "MyOp") as scope:
    a = tf.convert_to_tensor(a, name="a")
    b = tf.convert_to_tensor(b, name="b")
    c = tf.convert_to_tensor(c, name="c")
    # Define some computation that uses `a`, `b`, and `c`.
    return foo_op(..., name=scope)
```

**Args:**

* `values`: The list of `Tensor` arguments that are passed to the op function.
* `name`: The name argument that is passed to the op function.
* `default_name`: The default name to use if the `name` argument is `None`.

**Returns:**

A context manager for use in defining Python ops. Yields the name scope.

**Raises:**

* `ValueError`: if neither `name` nor `default_name` is provided.

***

### `tf.get_seed(op_seed)` <a href="#get_seed" id="get_seed"></a>

Returns the local seeds an operation should use given an op-specific seed.

Given operation-specific seed, `op_seed`, this helper function returns two seeds derived from graph-level and op-level seeds. Many random operations internally use the two seeds to allow user to change the seed globally for a graph, or for only specific operations.

For details on how the graph-level seed interacts with op seeds, see [`set_random_seed`](broken-reference).

**Args:**

* `op_seed`: integer.

**Returns:**

A tuple of two integers that should be used for the local seed of this operation.

## For libraries building on TensorFlow

***

### `tf.register_tensor_conversion_function(base_type, conversion_func, priority=100)` <a href="#register_tensor_conversion_function" id="register_tensor_conversion_function"></a>

Registers a function for converting objects of `base_type` to `Tensor`.

The conversion function must have the following signature:

```
def conversion_func(value, dtype=None, name=None, as_ref=False):
  # ...
```

It must return a `Tensor` with the given `dtype` if specified. If the conversion function creates a new `Tensor`, it should use the given `name` if specified. All exceptions will be propagated to the caller.

The conversion function may return `NotImplemented` for some inputs. In this case, the conversion process will continue to try subsequent conversion functions.

If `as_ref` is true, the function must return a `Tensor` reference, such as a `Variable`.

NOTE: The conversion functions will execute in order of priority, followed by order of registration. To ensure that a conversion function `F` runs before another conversion function `G`, ensure that `F` is registered with a smaller priority than `G`.

**Args:**

* `base_type`: The base type or tuple of base types for all objects that `conversion_func` accepts.
* `conversion_func`: A function that converts instances of `base_type` to `Tensor`.
* `priority`: Optional integer that indicates the priority for applying this conversion function. Conversion functions with smaller priority values run earlier than conversion functions with larger priority values. Defaults to 100.

**Raises:**

* `TypeError`: If the arguments do not have the appropriate type.

## Other Functions and Classes

***

### `class tf.DeviceSpec` <a href="#devicespec" id="devicespec"></a>

Represents a (possibly partial) specification for a TensorFlow device.

`DeviceSpec`s are used throughout TensorFlow to describe where state is stored and computations occur. Using `DeviceSpec` allows you to parse device spec strings to verify their validity, merge them or compose them programmatically.

Example:

```python
# Place the operations on device "GPU:0" in the "ps" job.
device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
with tf.device(device_spec):
  # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
  my_var = tf.Variable(..., name="my_variable")
  squared_var = tf.square(my_var)
```

If a `DeviceSpec` is partially specified, it will be merged with other `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec` components defined in inner scopes take precedence over those defined in outer scopes.

```python
with tf.device(DeviceSpec(job="train", )):
  with tf.device(DeviceSpec(job="ps", device_type="GPU", device_index=0):
    # Nodes created here will be assigned to /job:ps/device:GPU:0.
  with tf.device(DeviceSpec(device_type="GPU", device_index=1):
    # Nodes created here will be assigned to /job:train/device:GPU:1.
```

A `DeviceSpec` consists of 5 components -- each of which is optionally specified:

* Job: The job name.
* Replica: The replica index.
* Task: The task index.
* Device type: The device type string (e.g. "CPU" or "GPU").
* Device index: The device index.

***

#### `tf.DeviceSpec.__init__(job=None, replica=None, task=None, device_type=None, device_index=None)` {#DeviceSpec.**init**}

Create a new `DeviceSpec` object.

**Args:**

* `job`: string. Optional job name.
* `replica`: int. Optional replica index.
* `task`: int. Optional task index.
* `device_type`: Optional device type string (e.g. "CPU" or "GPU")
* `device_index`: int. Optional device index. If left unspecified, device represents 'any' device\_index.

***

#### `tf.DeviceSpec.from_string(spec)` <a href="#devicespec.from_string" id="devicespec.from_string"></a>

Construct a `DeviceSpec` from a string.

**Args:**

* `spec`: a string of the form /job:/replica:/task:/device:CPU: or /job:/replica:/task:/device:GPU: as cpu and gpu are mutually exclusive. All entries are optional.

**Returns:**

A DeviceSpec.

***

#### `tf.DeviceSpec.job` <a href="#devicespec.job" id="devicespec.job"></a>

***

#### `tf.DeviceSpec.merge_from(dev)` <a href="#devicespec.merge_from" id="devicespec.merge_from"></a>

Merge the properties of "dev" into this `DeviceSpec`.

**Args:**

* `dev`: a `DeviceSpec`.

***

#### `tf.DeviceSpec.parse_from_string(spec)` <a href="#devicespec.parse_from_string" id="devicespec.parse_from_string"></a>

Parse a `DeviceSpec` name into its components.

**Args:**

* `spec`: a string of the form /job:/replica:/task:/device:CPU: or /job:/replica:/task:/device:GPU: as cpu and gpu are mutually exclusive. All entries are optional.

**Returns:**

The `DeviceSpec`.

**Raises:**

* `ValueError`: if the spec was not valid.

***

#### `tf.DeviceSpec.replica` <a href="#devicespec.replica" id="devicespec.replica"></a>

***

#### `tf.DeviceSpec.task` <a href="#devicespec.task" id="devicespec.task"></a>

***

#### `tf.DeviceSpec.to_string()` <a href="#devicespec.to_string" id="devicespec.to_string"></a>

Return a string representation of this `DeviceSpec`.

**Returns:**

a string of the form /job:/replica:/task:/device:\<device\_type>:.

***

### `class tf.bytes` <a href="#bytes" id="bytes"></a>

str(object='') -> string

Return a nice string representation of the object. If the argument is a string, the return value is the same object.
