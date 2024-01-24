# class tensorflow::Tensor

Represents an n-dimensional array of values.

\###Member Details

#### `tensorflow::Tensor::Tensor()` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Default Tensor constructor. Creates a 1-dimension, 0-element float tensor.

#### `tensorflow::Tensor::Tensor(DataType type, const TensorShape &shape)` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Creates a Tensor of the given `type` and `shape`. If LogMemory::IsEnabled() the allocation is logged as coming from an unknown kernel and step. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate\_\* methods to allocate a new tensor, which record the kernel and step.

The underlying buffer is allocated using a `CPUAllocator`.

#### `tensorflow::Tensor::Tensor(Allocator *a, DataType type, const TensorShape &shape)` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Creates a tensor with the input `type` and `shape`, using the allocator `a` to allocate the underlying buffer. If LogMemory::IsEnabled() the allocation is logged as coming from an unknown kernel and step. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate\_\* methods to allocate a new tensor, which record the kernel and step.

`a` must outlive the lifetime of this Tensor .

#### `tensorflow::Tensor::Tensor(Allocator *a, DataType type, const TensorShape &shape, const AllocationAttributes &allocation_attr)` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Creates a tensor with the input `type` and `shape`, using the allocator `a` and the specified "allocation\_attr" to allocate the underlying buffer. If the kernel and step are known allocation\_attr.allocation\_will\_be\_logged should be set to true and LogMemory::RecordTensorAllocation should be called after the tensor is constructed. Calling the Tensor constructor directly from within an Op is deprecated: use the OpKernelConstruction/OpKernelContext allocate\_\* methods to allocate a new tensor, which record the kernel and step.

`a` must outlive the lifetime of this Tensor .

#### `tensorflow::Tensor::Tensor(DataType type)` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Creates an uninitialized Tensor of the given data type.

#### `tensorflow::Tensor::Tensor(const Tensor &other)` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

#### `tensorflow::Tensor::~Tensor()` <a href="#tensorflow_tensor_tensor" id="tensorflow_tensor_tensor"></a>

Copy constructor.

#### `DataType tensorflow::Tensor::dtype() const` <a href="#datatype_tensorflow_tensor_dtype" id="datatype_tensorflow_tensor_dtype"></a>

Returns the data type.

#### `const TensorShape& tensorflow::Tensor::shape() const` <a href="#const_tensorshape_tensorflow_tensor_shape" id="const_tensorshape_tensorflow_tensor_shape"></a>

Returns the shape of the tensor.

#### `int tensorflow::Tensor::dims() const` <a href="#int_tensorflow_tensor_dims" id="int_tensorflow_tensor_dims"></a>

Convenience accessor for the tensor shape.

For all shape accessors, see comments for relevant methods of `TensorShape` in `tensor_shape.h`.

#### `int64 tensorflow::Tensor::dim_size(int d) const` <a href="#int64_tensorflow_tensor_dim_size" id="int64_tensorflow_tensor_dim_size"></a>

Convenience accessor for the tensor shape.

#### `int64 tensorflow::Tensor::NumElements() const` <a href="#int64_tensorflow_tensor_numelements" id="int64_tensorflow_tensor_numelements"></a>

Convenience accessor for the tensor shape.

#### `bool tensorflow::Tensor::IsSameSize(const Tensor &b) const` <a href="#bool_tensorflow_tensor_issamesize" id="bool_tensorflow_tensor_issamesize"></a>

#### `bool tensorflow::Tensor::SharesBufferWith(const Tensor &b) const` <a href="#bool_tensorflow_tensor_sharesbufferwith" id="bool_tensorflow_tensor_sharesbufferwith"></a>

#### `size_t tensorflow::Tensor::BufferHash() const` <a href="#size_t_tensorflow_tensor_bufferhash" id="size_t_tensorflow_tensor_bufferhash"></a>

#### `bool tensorflow::Tensor::IsInitialized() const` <a href="#bool_tensorflow_tensor_isinitialized" id="bool_tensorflow_tensor_isinitialized"></a>

Has this Tensor been initialized?

#### `size_t tensorflow::Tensor::TotalBytes() const` <a href="#size_t_tensorflow_tensor_totalbytes" id="size_t_tensorflow_tensor_totalbytes"></a>

Returns the estimated memory usage of this tensor.

#### `bool tensorflow::Tensor::IsAligned() const` <a href="#bool_tensorflow_tensor_isaligned" id="bool_tensorflow_tensor_isaligned"></a>

Returns true iff this tensor is aligned.

#### `Tensor& tensorflow::Tensor::operator=(const Tensor &other)` <a href="#tensor_tensorflow_tensor_operator" id="tensor_tensorflow_tensor_operator"></a>

Assign operator. This tensor shares other's underlying storage.

#### `bool tensorflow::Tensor::CopyFrom(const Tensor &other, const TensorShape &shape) TF_MUST_USE_RESULT` <a href="#bool_tensorflow_tensor_copyfrom" id="bool_tensorflow_tensor_copyfrom"></a>

Copy the other tensor into this tensor and reshape it.

This tensor shares other's underlying storage. Returns `true` iff `other.shape()` has the same number of elements of the given `shape`.

#### `Tensor tensorflow::Tensor::Slice(int64 dim0_start, int64 dim0_limit) const` <a href="#tensor_tensorflow_tensor_slice" id="tensor_tensorflow_tensor_slice"></a>

Slice this tensor along the 1st dimension.

I.e., the returned tensor satisfies returned\[i, ...] == this\[dim0\_start + i, ...]. The returned tensor shares the underlying tensor buffer with this tensor.

NOTE: The returned tensor may not satisfies the same alignment requirement as this tensor depending on the shape. The caller must check the returned tensor's alignment before calling certain methods that have alignment requirement (e.g., `flat()`, `tensor()`).

REQUIRES: `dims()` >= 1 REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`

#### `bool tensorflow::Tensor::FromProto(const TensorProto &other) TF_MUST_USE_RESULT` <a href="#bool_tensorflow_tensor_fromproto" id="bool_tensorflow_tensor_fromproto"></a>

Parse `other` and construct the tensor.

Returns `true` iff the parsing succeeds. If the parsing fails, the state of `*this` is unchanged.

#### `bool tensorflow::Tensor::FromProto(Allocator *a, const TensorProto &other) TF_MUST_USE_RESULT` <a href="#bool_tensorflow_tensor_fromproto" id="bool_tensorflow_tensor_fromproto"></a>

#### `void tensorflow::Tensor::AsProtoField(TensorProto *proto) const` <a href="#void_tensorflow_tensor_asprotofield" id="void_tensorflow_tensor_asprotofield"></a>

Fills in `proto` with `*this` tensor's content.

`AsProtoField()` fills in the repeated field for `proto.dtype()`, while `AsProtoTensorContent()` encodes the content in `proto.tensor_content()` in a compact form.

#### `void tensorflow::Tensor::AsProtoTensorContent(TensorProto *proto) const` <a href="#void_tensorflow_tensor_asprototensorcontent" id="void_tensorflow_tensor_asprototensorcontent"></a>

#### `TTypes<T>::Vec tensorflow::Tensor::vec()` <a href="#ttypes_t_vec_tensorflow_tensor_vec" id="ttypes_t_vec_tensorflow_tensor_vec"></a>

Return the tensor data as an `Eigen::Tensor` with the type and sizes of this `Tensor`.

Use these methods when you know the data type and the number of dimensions of the Tensor and you want an `Eigen::Tensor` automatically sized to the `Tensor` sizes. The implementation check fails if either type or sizes mismatch.

Example:

```c++
Tensor my_mat(...built with Shape{rows: 3, cols: 5}...);
auto mat = my_mat.matrix<T>();    // 2D Eigen::Tensor, 3 x 5.
auto mat = my_mat.tensor<T, 2>(); // 2D Eigen::Tensor, 3 x 5.
auto vec = my_mat.vec<T>();       // CHECK fails as my_mat is 2D.
auto vec = my_mat.tensor<T, 3>(); // CHECK fails as my_mat is 2D.
auto mat = my_mat.matrix<int32>();// CHECK fails as type mismatch.

```

#### `TTypes<T>::Matrix tensorflow::Tensor::matrix()` <a href="#ttypes_t_matrix_tensorflow_tensor_matrix" id="ttypes_t_matrix_tensorflow_tensor_matrix"></a>

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::tensor()` <a href="#ttypes_t_ndims_tensor_tensorflow_tensor_tensor" id="ttypes_t_ndims_tensor_tensorflow_tensor_tensor"></a>

#### `TTypes<T>::Flat tensorflow::Tensor::flat()` <a href="#ttypes_t_flat_tensorflow_tensor_flat" id="ttypes_t_flat_tensorflow_tensor_flat"></a>

Return the tensor data as an `Eigen::Tensor` of the data type and a specified shape.

These methods allow you to access the data with the dimensions and sizes of your choice. You do not need to know the number of dimensions of the Tensor to call them. However, they `CHECK` that the type matches and the dimensions requested creates an `Eigen::Tensor` with the same number of elements as the tensor.

Example:

```c++
Tensor my_ten(...built with Shape{planes: 4, rows: 3, cols: 5}...);
// 1D Eigen::Tensor, size 60:
auto flat = my_ten.flat<T>();
// 2D Eigen::Tensor 12 x 5:
auto inner = my_ten.flat_inner_dims<T>();
// 2D Eigen::Tensor 4 x 15:
auto outer = my_ten.shaped<T, 2>({4, 15});
// CHECK fails, bad num elements:
auto outer = my_ten.shaped<T, 2>({4, 8});
// 3D Eigen::Tensor 6 x 5 x 2:
auto weird = my_ten.shaped<T, 3>({6, 5, 2});
// CHECK fails, type mismatch:
auto bad   = my_ten.flat<int32>();

```

#### `TTypes<T>::UnalignedFlat tensorflow::Tensor::unaligned_flat()` <a href="#ttypes_t_unalignedflat_tensorflow_tensor_unaligned_flat" id="ttypes_t_unalignedflat_tensorflow_tensor_unaligned_flat"></a>

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::flat_inner_dims()` <a href="#ttypes_t_ndims_tensor_tensorflow_tensor_flat_inner_dims" id="ttypes_t_ndims_tensor_tensorflow_tensor_flat_inner_dims"></a>

Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all Tensor dimensions but the last NDIMS-1 into the first dimension of the result. If NDIMS > dims() then leading dimensions of size 1 will be added to make the output rank NDIMS.

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::flat_outer_dims()` <a href="#ttypes_t_ndims_tensor_tensorflow_tensor_flat_outer_dims" id="ttypes_t_ndims_tensor_tensorflow_tensor_flat_outer_dims"></a>

Returns the data as an Eigen::Tensor with NDIMS dimensions, collapsing all Tensor dimensions but the first NDIMS-1 into the last dimension of the result. If NDIMS > dims() then trailing dimensions of size 1 will be added to make the output rank NDIMS.

#### `TTypes< T, NDIMS >::Tensor tensorflow::Tensor::shaped(gtl::ArraySlice< int64 > new_sizes)` <a href="#ttypes_t_ndims_tensor_tensorflow_tensor_shaped" id="ttypes_t_ndims_tensor_tensorflow_tensor_shaped"></a>

#### `TTypes< T, NDIMS >::UnalignedTensor tensorflow::Tensor::unaligned_shaped(gtl::ArraySlice< int64 > new_sizes)` <a href="#ttypes_t_ndims_unalignedtensor_tensorflow_tensor_unaligned_shaped" id="ttypes_t_ndims_unalignedtensor_tensorflow_tensor_unaligned_shaped"></a>

#### `TTypes< T >::Scalar tensorflow::Tensor::scalar()` <a href="#ttypes_t_scalar_tensorflow_tensor_scalar" id="ttypes_t_scalar_tensorflow_tensor_scalar"></a>

Return the Tensor data as a `TensorMap` of fixed size 1: `TensorMap<TensorFixedSize<T, 1>>`.

Using `scalar()` allows the compiler to perform optimizations as the size of the tensor is known at compile time.

#### `TTypes<T>::ConstVec tensorflow::Tensor::vec() const` <a href="#ttypes_t_constvec_tensorflow_tensor_vec" id="ttypes_t_constvec_tensorflow_tensor_vec"></a>

Const versions of all the methods above.

#### `TTypes<T>::ConstMatrix tensorflow::Tensor::matrix() const` <a href="#ttypes_t_constmatrix_tensorflow_tensor_matrix" id="ttypes_t_constmatrix_tensorflow_tensor_matrix"></a>

#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::tensor() const` <a href="#ttypes_t_ndims_consttensor_tensorflow_tensor_tensor" id="ttypes_t_ndims_consttensor_tensorflow_tensor_tensor"></a>

#### `TTypes<T>::ConstFlat tensorflow::Tensor::flat() const` <a href="#ttypes_t_constflat_tensorflow_tensor_flat" id="ttypes_t_constflat_tensorflow_tensor_flat"></a>

#### `TTypes<T>::UnalignedConstFlat tensorflow::Tensor::unaligned_flat() const` <a href="#ttypes_t_unalignedconstflat_tensorflow_tensor_unaligned_flat" id="ttypes_t_unalignedconstflat_tensorflow_tensor_unaligned_flat"></a>

#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::shaped(gtl::ArraySlice< int64 > new_sizes) const` <a href="#ttypes_t_ndims_consttensor_tensorflow_tensor_shaped" id="ttypes_t_ndims_consttensor_tensorflow_tensor_shaped"></a>

#### `TTypes< T, NDIMS >::UnalignedConstTensor tensorflow::Tensor::unaligned_shaped(gtl::ArraySlice< int64 > new_sizes) const` <a href="#ttypes_t_ndims_unalignedconsttensor_tensorflow_tensor_unaligned_shaped" id="ttypes_t_ndims_unalignedconsttensor_tensorflow_tensor_unaligned_shaped"></a>

#### `TTypes< T >::ConstScalar tensorflow::Tensor::scalar() const` <a href="#ttypes_t_constscalar_tensorflow_tensor_scalar" id="ttypes_t_constscalar_tensorflow_tensor_scalar"></a>

#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::flat_inner_dims() const` <a href="#ttypes_t_ndims_consttensor_tensorflow_tensor_flat_inner_dims" id="ttypes_t_ndims_consttensor_tensorflow_tensor_flat_inner_dims"></a>

#### `TTypes< T, NDIMS >::ConstTensor tensorflow::Tensor::flat_outer_dims() const` <a href="#ttypes_t_ndims_consttensor_tensorflow_tensor_flat_outer_dims" id="ttypes_t_ndims_consttensor_tensorflow_tensor_flat_outer_dims"></a>

#### `string tensorflow::Tensor::SummarizeValue(int64 max_entries) const` <a href="#string_tensorflow_tensor_summarizevalue" id="string_tensorflow_tensor_summarizevalue"></a>

Render the first `max_entries` values in `*this` into a string.

#### `string tensorflow::Tensor::DebugString() const` <a href="#string_tensorflow_tensor_debugstring" id="string_tensorflow_tensor_debugstring"></a>

A human-readable summary of the tensor suitable for debugging.

#### `void tensorflow::Tensor::FillDescription(TensorDescription *description) const` <a href="#void_tensorflow_tensor_filldescription" id="void_tensorflow_tensor_filldescription"></a>

Fill in the `TensorDescription` proto with metadata about the tensor that is useful for monitoring and debugging.

#### `StringPiece tensorflow::Tensor::tensor_data() const` <a href="#stringpiece_tensorflow_tensor_tensor_data" id="stringpiece_tensorflow_tensor_tensor_data"></a>

Returns a `StringPiece` mapping the current tensor's buffer.

The returned `StringPiece` may point to memory location on devices that the CPU cannot address directly.

NOTE: The underlying tensor buffer is refcounted, so the lifetime of the contents mapped by the `StringPiece` matches the lifetime of the buffer; callers should arrange to make sure the buffer does not get destroyed while the `StringPiece` is still used.

REQUIRES: `DataTypeCanUseMemcpy( dtype() )`.

#### `void tensorflow::Tensor::UnsafeCopyFromInternal(const Tensor &, const TensorShape &)` <a href="#void_tensorflow_tensor_unsafecopyfrominternal" id="void_tensorflow_tensor_unsafecopyfrominternal"></a>

Copy the other tensor into this tensor and reshape it and reinterpret the buffer's datatype.

This tensor shares other's underlying storage.
