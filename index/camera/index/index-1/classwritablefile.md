# class tensorflow::WritableFile

A file abstraction for sequential writing.

The implementation must provide buffering since callers may append small fragments at a time to the file.

\###Member Details

#### `tensorflow::WritableFile::WritableFile()` <a href="#tensorflow_writablefile_writablefile" id="tensorflow_writablefile_writablefile"></a>

#### `tensorflow::WritableFile::~WritableFile()` <a href="#tensorflow_writablefile_writablefile" id="tensorflow_writablefile_writablefile"></a>

#### `virtual Status tensorflow::WritableFile::Append(const StringPiece &data)=0` <a href="#virtual_status_tensorflow_writablefile_append" id="virtual_status_tensorflow_writablefile_append"></a>

#### `virtual Status tensorflow::WritableFile::Close()=0` <a href="#virtual_status_tensorflow_writablefile_close" id="virtual_status_tensorflow_writablefile_close"></a>

#### `virtual Status tensorflow::WritableFile::Flush()=0` <a href="#virtual_status_tensorflow_writablefile_flush" id="virtual_status_tensorflow_writablefile_flush"></a>

#### `virtual Status tensorflow::WritableFile::Sync()=0` <a href="#virtual_status_tensorflow_writablefile_sync" id="virtual_status_tensorflow_writablefile_sync"></a>
