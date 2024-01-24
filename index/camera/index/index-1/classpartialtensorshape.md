# class tensorflow::PartialTensorShape

Manages the partially known dimensions of a Tensor and their sizes.

\###Member Details

#### `tensorflow::PartialTensorShape::PartialTensorShape()` <a href="#tensorflow_partialtensorshape_partialtensorshape" id="tensorflow_partialtensorshape_partialtensorshape"></a>

Construct an unknown `PartialTensorShape`.

#### `tensorflow::PartialTensorShape::PartialTensorShape(gtl::ArraySlice< int64 > dim_sizes)` <a href="#tensorflow_partialtensorshape_partialtensorshape" id="tensorflow_partialtensorshape_partialtensorshape"></a>

Construct a `PartialTensorShape` from the provided sizes. REQUIRES: `dim_sizes[i] >= 0`

#### `tensorflow::PartialTensorShape::PartialTensorShape(std::initializer_list< int64 > dim_sizes)` <a href="#tensorflow_partialtensorshape_partialtensorshape" id="tensorflow_partialtensorshape_partialtensorshape"></a>

#### `tensorflow::PartialTensorShape::PartialTensorShape(const TensorShapeProto &proto)` <a href="#tensorflow_partialtensorshape_partialtensorshape" id="tensorflow_partialtensorshape_partialtensorshape"></a>

REQUIRES: `IsValid(proto)`

#### `PartialTensorShape tensorflow::PartialTensorShape::Concatenate(int64 size) const` <a href="#partialtensorshape_tensorflow_partialtensorshape_concatenate" id="partialtensorshape_tensorflow_partialtensorshape_concatenate"></a>

Add a dimension to the end ("inner-most"), returns a new PartialTensorShape . REQUIRES: `size >= -1`, where -1 means unknown

#### `PartialTensorShape tensorflow::PartialTensorShape::Concatenate(const PartialTensorShape &shape) const` <a href="#partialtensorshape_tensorflow_partialtensorshape_concatenate" id="partialtensorshape_tensorflow_partialtensorshape_concatenate"></a>

Appends all the dimensions from `shape`. Returns a new PartialTensorShape .

#### `Status tensorflow::PartialTensorShape::MergeWith(const PartialTensorShape &shape, PartialTensorShape *result) const` <a href="#status_tensorflow_partialtensorshape_mergewith" id="status_tensorflow_partialtensorshape_mergewith"></a>

Merges all the dimensions from `shape`. Returns `InvalidArgument` error if either `shape` has a different rank or if any of the dimensions are incompatible.

#### `int tensorflow::PartialTensorShape::dims() const` <a href="#int_tensorflow_partialtensorshape_dims" id="int_tensorflow_partialtensorshape_dims"></a>

Return the number of dimensions in the tensor. If the number of dimensions is unknown, return -1.

#### `bool tensorflow::PartialTensorShape::IsFullyDefined() const` <a href="#bool_tensorflow_partialtensorshape_isfullydefined" id="bool_tensorflow_partialtensorshape_isfullydefined"></a>

Return true iff the rank and all of the dimensions are well defined.

#### `bool tensorflow::PartialTensorShape::IsCompatibleWith(const PartialTensorShape &shape) const` <a href="#bool_tensorflow_partialtensorshape_iscompatiblewith" id="bool_tensorflow_partialtensorshape_iscompatiblewith"></a>

Return true iff the ranks match, and if the dimensions all either match or one is unknown.

#### `bool tensorflow::PartialTensorShape::IsCompatibleWith(const TensorShape &shape) const` <a href="#bool_tensorflow_partialtensorshape_iscompatiblewith" id="bool_tensorflow_partialtensorshape_iscompatiblewith"></a>

Return true iff the dimensions of `shape` are compatible with `*this`.

#### `int64 tensorflow::PartialTensorShape::dim_size(int d) const` <a href="#int64_tensorflow_partialtensorshape_dim_size" id="int64_tensorflow_partialtensorshape_dim_size"></a>

Returns the number of elements in dimension `d`. REQUIRES: `0 <= d < dims()`

#### `gtl::ArraySlice<int64> tensorflow::PartialTensorShape::dim_sizes() const` <a href="#gtl_arrayslice_int64_tensorflow_partialtensorshape_dim_sizes" id="gtl_arrayslice_int64_tensorflow_partialtensorshape_dim_sizes"></a>

Returns sizes of all dimensions.

#### `void tensorflow::PartialTensorShape::AsProto(TensorShapeProto *proto) const` <a href="#void_tensorflow_partialtensorshape_asproto" id="void_tensorflow_partialtensorshape_asproto"></a>

Fill `*proto` from `*this`.

#### `bool tensorflow::PartialTensorShape::AsTensorShape(TensorShape *tensor_shape) const` <a href="#bool_tensorflow_partialtensorshape_astensorshape" id="bool_tensorflow_partialtensorshape_astensorshape"></a>

#### `string tensorflow::PartialTensorShape::DebugString() const` <a href="#string_tensorflow_partialtensorshape_debugstring" id="string_tensorflow_partialtensorshape_debugstring"></a>

For error messages.

#### `bool tensorflow::PartialTensorShape::IsValid(const TensorShapeProto &proto)` <a href="#bool_tensorflow_partialtensorshape_isvalid" id="bool_tensorflow_partialtensorshape_isvalid"></a>

Returns `true` iff `proto` is a valid partial tensor shape.

#### `Status tensorflow::PartialTensorShape::IsValidShape(const TensorShapeProto &proto)` <a href="#status_tensorflow_partialtensorshape_isvalidshape" id="status_tensorflow_partialtensorshape_isvalidshape"></a>

Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error status otherwise.

#### `string tensorflow::PartialTensorShape::DebugString(const TensorShapeProto &proto)` <a href="#string_tensorflow_partialtensorshape_debugstring" id="string_tensorflow_partialtensorshape_debugstring"></a>

#### `static Status tensorflow::PartialTensorShape::MakePartialShape(const int32 *dims, int n, PartialTensorShape *out)` <a href="#static_status_tensorflow_partialtensorshape_makepartialshape" id="static_status_tensorflow_partialtensorshape_makepartialshape"></a>

Returns a `PartialTensorShape` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`. Values of -1 are considered "unknown".

#### `static Status tensorflow::PartialTensorShape::MakePartialShape(const int64 *dims, int n, PartialTensorShape *out)` <a href="#static_status_tensorflow_partialtensorshape_makepartialshape" id="static_status_tensorflow_partialtensorshape_makepartialshape"></a>
