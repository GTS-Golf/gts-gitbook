# class tensorflow::Session

A Session instance lets a caller drive a TensorFlow graph computation.

When a Session is created with a given target, a new Session object is bound to the universe of resources specified by that target. Those resources are available to this session to perform computation described in the GraphDef. After extending the session with a graph, the caller uses the Run() API to perform the computation and potentially fetch outputs as Tensors.

Example:

```c++
// ... Create or load graph into "graph".

// This example uses the default options which connects
// to a local runtime.
tensorflow::SessionOptions options;
std::unique_ptr<tensorflow::Session>
session(tensorflow::NewSession(options));

// Create the session with this graph.
tensorflow::Status s = session->Create(graph);
if (!s.ok()) { ... }

// Run the graph and fetch the first output of the "output"
// operation, and also run to but do not return anything
// for the "update_state" operation.
std::vector<tensorflow::Tensor> outputs;
s = session->Run({}, {"output:0"}, {"update_state"}, &outputs);
if (!s.ok()) { ... }

// Map the output as a flattened float tensor, and do something
// with it.
auto output_tensor = outputs[0].flat<float>();
if (output_tensor(0) > 0.5) { ... }

// Close the session to release the resources associated with
// this session.
session->Close();

```

A Session allows concurrent calls to Run() , though a Session must be created / extended by a single thread.

Only one thread must call Close() , and Close() must only be called after all other calls to Run() have returned.

\###Member Details

#### `tensorflow::Session::Session()` <a href="#tensorflow_session_session" id="tensorflow_session_session"></a>

#### `virtual tensorflow::Session::~Session()` <a href="#virtual_tensorflow_session_session" id="virtual_tensorflow_session_session"></a>

#### `virtual Status tensorflow::Session::Create(const GraphDef &graph)=0` <a href="#virtual_status_tensorflow_session_create" id="virtual_status_tensorflow_session_create"></a>

Create the graph to be used for the session.

Returns an error if this session has already been created with a graph. To re-use the session with a different graph, the caller must Close() the session first.

#### `virtual Status tensorflow::Session::Extend(const GraphDef &graph)=0` <a href="#virtual_status_tensorflow_session_extend" id="virtual_status_tensorflow_session_extend"></a>

Adds operations to the graph that is already registered with the Session .

The names of new operations in "graph" must not exist in the graph that is already registered.

#### `virtual Status tensorflow::Session::Run(const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names, const std::vector< string > &target_node_names, std::vector< Tensor > *outputs)=0` <a href="#virtual_status_tensorflow_session_run" id="virtual_status_tensorflow_session_run"></a>

Runs the graph with the provided input tensors and fills `outputs` for the endpoints specified in `output_tensor_names`. Runs to but does not return Tensors for the nodes in `target_node_names`.

The order of tensors in `outputs` will match the order provided by `output_tensor_names`.

If `Run` returns `OK()`, then `outputs->size()` will be equal to `output_tensor_names.size()`. If `Run` does not return `OK()`, the state of `outputs` is undefined.

REQUIRES: The name of each Tensor of the input or output must match a "Tensor endpoint" in the `GraphDef` passed to `Create()`.

REQUIRES: At least one of `output_tensor_names` and `target_node_names` must be non-empty.

REQUIRES: outputs is not nullptr if `output_tensor_names` is non-empty.

#### `virtual Status tensorflow::Session::Create(const RunOptions &run_options, const GraphDef &graph)` <a href="#virtual_status_tensorflow_session_create" id="virtual_status_tensorflow_session_create"></a>

Implementations which support `RunOptions`.

NOTE: This API is still experimental and may change.

#### `virtual Status tensorflow::Session::Extend(const RunOptions &run_options, const GraphDef &graph)` <a href="#virtual_status_tensorflow_session_extend" id="virtual_status_tensorflow_session_extend"></a>

#### `virtual Status tensorflow::Session::Close(const RunOptions &run_options)` <a href="#virtual_status_tensorflow_session_close" id="virtual_status_tensorflow_session_close"></a>

#### `virtual Status tensorflow::Session::Run(const RunOptions &run_options, const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_tensor_names, const std::vector< string > &target_node_names, std::vector< Tensor > *outputs, RunMetadata *run_metadata)` <a href="#virtual_status_tensorflow_session_run" id="virtual_status_tensorflow_session_run"></a>

Like `Run`, but allows users to pass in a `RunOptions` proto and to retrieve non-Tensor metadata output via a `RunMetadata` proto for this step. `run_metadata` may be nullptr, in which case any metadata output is discarded. NOTE: This API is still experimental and may change.

#### `virtual Status tensorflow::Session::PRunSetup(const std::vector< string > &input_names, const std::vector< string > &output_names, const std::vector< string > &target_nodes, string *handle)` <a href="#virtual_status_tensorflow_session_prunsetup" id="virtual_status_tensorflow_session_prunsetup"></a>

Sets up a graph for partial execution. All future feeds and fetches are specified by `input_names` and `output_names`. Returns `handle` that can be used to perform a sequence of partial feeds and fetches. NOTE: This API is still experimental and may change.

#### `virtual Status tensorflow::Session::PRun(const string &handle, const std::vector< std::pair< string, Tensor > > &inputs, const std::vector< string > &output_names, std::vector< Tensor > *outputs)` <a href="#virtual_status_tensorflow_session_prun" id="virtual_status_tensorflow_session_prun"></a>

Continues the pending execution specified by `handle` with the provided input tensors and fills `outputs` for the endpoints specified in `output_names`. NOTE: This API is still experimental and may change.

#### `virtual Status tensorflow::Session::Close()=0` <a href="#virtual_status_tensorflow_session_close" id="virtual_status_tensorflow_session_close"></a>

Closes this session.

Closing a session releases the resources used by this session on the TensorFlow runtime (specified during session creation by the `SessionOptions::target` field).
