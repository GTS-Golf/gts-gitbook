# class tensorflow::TensorShapeUtils

Static helper routines for `TensorShape`. Includes a few common predicates on a tensor shape.

\###Member Details

#### `static bool tensorflow::TensorShapeUtils::IsScalar(const TensorShape &shape)` <a href="#static_bool_tensorflow_tensorshapeutils_isscalar" id="static_bool_tensorflow_tensorshapeutils_isscalar"></a>

#### `static bool tensorflow::TensorShapeUtils::IsVector(const TensorShape &shape)` <a href="#static_bool_tensorflow_tensorshapeutils_isvector" id="static_bool_tensorflow_tensorshapeutils_isvector"></a>

#### `static bool tensorflow::TensorShapeUtils::IsVectorOrHigher(const TensorShape &shape)` <a href="#static_bool_tensorflow_tensorshapeutils_isvectororhigher" id="static_bool_tensorflow_tensorshapeutils_isvectororhigher"></a>

#### `static bool tensorflow::TensorShapeUtils::IsMatrix(const TensorShape &shape)` <a href="#static_bool_tensorflow_tensorshapeutils_ismatrix" id="static_bool_tensorflow_tensorshapeutils_ismatrix"></a>

#### `static bool tensorflow::TensorShapeUtils::IsMatrixOrHigher(const TensorShape &shape)` <a href="#static_bool_tensorflow_tensorshapeutils_ismatrixorhigher" id="static_bool_tensorflow_tensorshapeutils_ismatrixorhigher"></a>

#### `static Status tensorflow::TensorShapeUtils::MakeShape(const int32 *dims, int n, TensorShape *out)` <a href="#static_status_tensorflow_tensorshapeutils_makeshape" id="static_status_tensorflow_tensorshapeutils_makeshape"></a>

Returns a `TensorShape` whose dimensions are `dims[0]`, `dims[1]`, ..., `dims[n-1]`.

#### `static Status tensorflow::TensorShapeUtils::MakeShape(const int64 *dims, int n, TensorShape *out)` <a href="#static_status_tensorflow_tensorshapeutils_makeshape" id="static_status_tensorflow_tensorshapeutils_makeshape"></a>

#### `string tensorflow::TensorShapeUtils::ShapeListString(const gtl::ArraySlice< TensorShape > &shapes)` <a href="#string_tensorflow_tensorshapeutils_shapeliststring" id="string_tensorflow_tensorshapeutils_shapeliststring"></a>

#### `bool tensorflow::TensorShapeUtils::StartsWith(const TensorShape &shape0, const TensorShape &shape1)` <a href="#bool_tensorflow_tensorshapeutils_startswith" id="bool_tensorflow_tensorshapeutils_startswith"></a>
