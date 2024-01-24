# class tensorflow::TensorShape

\###Member Details

#### `uint8 tensorflow::TensorShape::buf[16][16]` <a href="#uint8_tensorflow_tensorshape_buf_16" id="uint8_tensorflow_tensorshape_buf_16"></a>

#### `Rep64* tensorflow::TensorShape::unused_aligner` <a href="#rep64_tensorflow_tensorshape_unused_aligner" id="rep64_tensorflow_tensorshape_unused_aligner"></a>

#### `tensorflow::TensorShape::TensorShape(gtl::ArraySlice< int64 > dim_sizes)` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

Construct a `TensorShape` from the provided sizes. REQUIRES: `dim_sizes[i] >= 0`

#### `tensorflow::TensorShape::TensorShape(std::initializer_list< int64 > dim_sizes)` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

#### `tensorflow::TensorShape::TensorShape(const TensorShapeProto &proto)` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

REQUIRES: `IsValid(proto)`

#### `tensorflow::TensorShape::TensorShape()` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

Create a tensor shape with no dimensions and one element, which you can then call `AddDim()` on.

#### `tensorflow::TensorShape::~TensorShape()` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

#### `tensorflow::TensorShape::TensorShape(const TensorShape &b)` <a href="#tensorflow_tensorshape_tensorshape" id="tensorflow_tensorshape_tensorshape"></a>

Copy the specified shape.

#### `void tensorflow::TensorShape::operator=(const TensorShape &b)` <a href="#void_tensorflow_tensorshape_operator" id="void_tensorflow_tensorshape_operator"></a>

#### `void tensorflow::TensorShape::Clear()` <a href="#void_tensorflow_tensorshape_clear" id="void_tensorflow_tensorshape_clear"></a>

Clear a tensor shape.

#### `void tensorflow::TensorShape::AddDim(int64 size)` <a href="#void_tensorflow_tensorshape_adddim" id="void_tensorflow_tensorshape_adddim"></a>

Add a dimension to the end ("inner-most"). REQUIRES: `size >= 0`

#### `void tensorflow::TensorShape::AppendShape(const TensorShape &shape)` <a href="#void_tensorflow_tensorshape_appendshape" id="void_tensorflow_tensorshape_appendshape"></a>

Appends all the dimensions from `shape`.

#### `void tensorflow::TensorShape::InsertDim(int d, int64 size)` <a href="#void_tensorflow_tensorshape_insertdim" id="void_tensorflow_tensorshape_insertdim"></a>

Insert a dimension somewhere in the `TensorShape`. REQUIRES: `0 <= d <= dims()` REQUIRES: `size >= 0`

#### `void tensorflow::TensorShape::set_dim(int d, int64 size)` <a href="#void_tensorflow_tensorshape_set_dim" id="void_tensorflow_tensorshape_set_dim"></a>

Modifies the size of the dimension `d` to be `size` REQUIRES: `0 <= d < dims()` REQUIRES: `size >= 0`

#### `void tensorflow::TensorShape::RemoveDim(int d)` <a href="#void_tensorflow_tensorshape_removedim" id="void_tensorflow_tensorshape_removedim"></a>

Removes dimension `d` from the `TensorShape`. REQUIRES: `0 <= d < dims()`

#### `int tensorflow::TensorShape::dims() const` <a href="#int_tensorflow_tensorshape_dims" id="int_tensorflow_tensorshape_dims"></a>

Return the number of dimensions in the tensor.

#### `int64 tensorflow::TensorShape::dim_size(int d) const` <a href="#int64_tensorflow_tensorshape_dim_size" id="int64_tensorflow_tensorshape_dim_size"></a>

Returns the number of elements in dimension `d`. REQUIRES: `0 <= d < dims()`

#### `gtl::InlinedVector< int64, 4 > tensorflow::TensorShape::dim_sizes() const` <a href="#gtl_inlinedvector_int64_4_tensorflow_tensorshape_dim_sizes" id="gtl_inlinedvector_int64_4_tensorflow_tensorshape_dim_sizes"></a>

Returns sizes of all dimensions.

#### `int64 tensorflow::TensorShape::num_elements() const` <a href="#int64_tensorflow_tensorshape_num_elements" id="int64_tensorflow_tensorshape_num_elements"></a>

Returns the number of elements in the tensor.

We use `int64` and not `size_t` to be compatible with `Eigen::Tensor` which uses `ptrdiff_t`.

#### `bool tensorflow::TensorShape::IsSameSize(const TensorShape &b) const` <a href="#bool_tensorflow_tensorshape_issamesize" id="bool_tensorflow_tensorshape_issamesize"></a>

Returns true if `*this` and `b` have the same sizes. Ignores dimension names.

#### `bool tensorflow::TensorShape::operator==(const TensorShape &b) const` <a href="#bool_tensorflow_tensorshape_operator" id="bool_tensorflow_tensorshape_operator"></a>

#### `bool tensorflow::TensorShape::operator!=(const TensorShape &b) const` <a href="#bool_tensorflow_tensorshape_operator" id="bool_tensorflow_tensorshape_operator"></a>

#### `void tensorflow::TensorShape::AsProto(TensorShapeProto *proto) const` <a href="#void_tensorflow_tensorshape_asproto" id="void_tensorflow_tensorshape_asproto"></a>

Fill `*proto` from `*this`.

#### `Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizes() const` <a href="#eigen_dsizes_eigen_denseindex_ndims_tensorflow_tensorshape_aseigendsizes" id="eigen_dsizes_eigen_denseindex_ndims_tensorflow_tensorshape_aseigendsizes"></a>

Fill `*dsizes` from `*this`.

#### `Eigen::DSizes< Eigen::DenseIndex, NDIMS > tensorflow::TensorShape::AsEigenDSizesWithPadding() const` <a href="#eigen_dsizes_eigen_denseindex_ndims_tensorflow_tensorshape_aseigendsizeswithpadding" id="eigen_dsizes_eigen_denseindex_ndims_tensorflow_tensorshape_aseigendsizeswithpadding"></a>

Same as `AsEigenDSizes()` but allows for `NDIMS > dims()` in which case we pad the rest of the sizes with 1.

#### `TensorShapeIter tensorflow::TensorShape::begin() const` <a href="#tensorshapeiter_tensorflow_tensorshape_begin" id="tensorshapeiter_tensorflow_tensorshape_begin"></a>

For iterating through the dimensions.

#### `TensorShapeIter tensorflow::TensorShape::end() const` <a href="#tensorshapeiter_tensorflow_tensorshape_end" id="tensorshapeiter_tensorflow_tensorshape_end"></a>

#### `string tensorflow::TensorShape::DebugString() const` <a href="#string_tensorflow_tensorshape_debugstring" id="string_tensorflow_tensorshape_debugstring"></a>

For error messages.

#### `void tensorflow::TensorShape::DumpRep() const` <a href="#void_tensorflow_tensorshape_dumprep" id="void_tensorflow_tensorshape_dumprep"></a>

#### `bool tensorflow::TensorShape::IsValid(const TensorShapeProto &proto)` <a href="#bool_tensorflow_tensorshape_isvalid" id="bool_tensorflow_tensorshape_isvalid"></a>

Returns `true` iff `proto` is a valid tensor shape.

#### `Status tensorflow::TensorShape::IsValidShape(const TensorShapeProto &proto)` <a href="#status_tensorflow_tensorshape_isvalidshape" id="status_tensorflow_tensorshape_isvalidshape"></a>

Returns `OK` iff `proto` is a valid tensor shape, and a descriptive error status otherwise.

#### `static constexpr int tensorflow::TensorShape::MaxDimensions()` <a href="#static_constexpr_int_tensorflow_tensorshape_maxdimensions" id="static_constexpr_int_tensorflow_tensorshape_maxdimensions"></a>

#### `string tensorflow::TensorShape::DebugString(const TensorShapeProto &proto)` <a href="#string_tensorflow_tensorshape_debugstring" id="string_tensorflow_tensorshape_debugstring"></a>

Same as `TensorShape(proto). DebugString()` but doesn't crash for invalid protos.
