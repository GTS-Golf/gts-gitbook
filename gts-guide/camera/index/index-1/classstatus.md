# class tensorflow::Status

\###Member Details

#### `tensorflow::Status::Status()` <a href="#tensorflow_status_status" id="tensorflow_status_status"></a>

Create a success status.

#### `tensorflow::Status::~Status()` <a href="#tensorflow_status_status" id="tensorflow_status_status"></a>

#### `tensorflow::Status::Status(tensorflow::error::Code code, tensorflow::StringPiece msg)` <a href="#tensorflow_status_status" id="tensorflow_status_status"></a>

Create a status with the specified error code and msg as a human-readable string containing more detailed information.

#### `tensorflow::Status::Status(const Status &s)` <a href="#tensorflow_status_status" id="tensorflow_status_status"></a>

Copy the specified status.

#### `void tensorflow::Status::operator=(const Status &s)` <a href="#void_tensorflow_status_operator" id="void_tensorflow_status_operator"></a>

#### `bool tensorflow::Status::ok() const` <a href="#bool_tensorflow_status_ok" id="bool_tensorflow_status_ok"></a>

Returns true iff the status indicates success.

#### `tensorflow::error::Code tensorflow::Status::code() const` <a href="#tensorflow_error_code_tensorflow_status_code" id="tensorflow_error_code_tensorflow_status_code"></a>

#### `const string& tensorflow::Status::error_message() const` <a href="#const_string_tensorflow_status_error_message" id="const_string_tensorflow_status_error_message"></a>

#### `bool tensorflow::Status::operator==(const Status &x) const` <a href="#bool_tensorflow_status_operator" id="bool_tensorflow_status_operator"></a>

#### `bool tensorflow::Status::operator!=(const Status &x) const` <a href="#bool_tensorflow_status_operator" id="bool_tensorflow_status_operator"></a>

#### `void tensorflow::Status::Update(const Status &new_status)` <a href="#void_tensorflow_status_update" id="void_tensorflow_status_update"></a>

If `ok()`, stores `new_status` into `*this`. If `!ok()`, preserves the current status, but may augment with additional information about `new_status`.

Convenient way of keeping track of the first error encountered. Instead of: `if (overall_status.ok()) overall_status = new_status` Use: `overall_status.Update(new_status);`

#### `string tensorflow::Status::ToString() const` <a href="#string_tensorflow_status_tostring" id="string_tensorflow_status_tostring"></a>

Return a string representation of this status suitable for printing. Returns the string `"OK"` for success.

#### `return tensorflow::Status::OK()` <a href="#return_tensorflow_status_ok" id="return_tensorflow_status_ok"></a>
