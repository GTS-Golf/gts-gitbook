# class tensorflow::Env

An interface used by the tensorflow implementation to access operating system functionality like the filesystem etc.

Callers may wish to provide a custom Env object to get fine grain control.

All Env implementations are safe for concurrent access from multiple threads without any external synchronization.

\###Member Details

#### `tensorflow::Env::Env()` <a href="#tensorflow_env_env" id="tensorflow_env_env"></a>

#### `virtual tensorflow::Env::~Env()=default` <a href="#virtual_tensorflow_env_env" id="virtual_tensorflow_env_env"></a>

#### `Status tensorflow::Env::GetFileSystemForFile(const string &fname, FileSystem **result)` <a href="#status_tensorflow_env_getfilesystemforfile" id="status_tensorflow_env_getfilesystemforfile"></a>

Returns the FileSystem object to handle operations on the file specified by 'fname'. The FileSystem object is used as the implementation for the file system related (non-virtual) functions that follow. Returned FileSystem object is still owned by the Env object and will.

#### `Status tensorflow::Env::GetRegisteredFileSystemSchemes(std::vector< string > *schemes)` <a href="#status_tensorflow_env_getregisteredfilesystemschemes" id="status_tensorflow_env_getregisteredfilesystemschemes"></a>

Returns the file system schemes registered for this Env .

#### `Status tensorflow::Env::RegisterFileSystem(const string &scheme, FileSystemRegistry::Factory factory)` <a href="#status_tensorflow_env_registerfilesystem" id="status_tensorflow_env_registerfilesystem"></a>

#### `Status tensorflow::Env::NewRandomAccessFile(const string &fname, RandomAccessFile **result)` <a href="#status_tensorflow_env_newrandomaccessfile" id="status_tensorflow_env_newrandomaccessfile"></a>

Creates a brand new random access read-only file with the specified name.

On success, stores a pointer to the new file in \*result and returns OK. On failure stores NULL in \*result and returns non-OK. If the file does not exist, returns a non-OK status.

The returned file may be concurrently accessed by multiple threads.

The ownership of the returned RandomAccessFile is passed to the caller and the object should be deleted when is not used. The file object shouldn't live longer than the Env object.

#### `Status tensorflow::Env::NewWritableFile(const string &fname, WritableFile **result)` <a href="#status_tensorflow_env_newwritablefile" id="status_tensorflow_env_newwritablefile"></a>

Creates an object that writes to a new file with the specified name.

Deletes any existing file with the same name and creates a new file. On success, stores a pointer to the new file in \*result and returns OK. On failure stores NULL in \*result and returns non-OK.

The returned file will only be accessed by one thread at a time.

The ownership of the returned WritableFile is passed to the caller and the object should be deleted when is not used. The file object shouldn't live longer than the Env object.

#### `Status tensorflow::Env::NewAppendableFile(const string &fname, WritableFile **result)` <a href="#status_tensorflow_env_newappendablefile" id="status_tensorflow_env_newappendablefile"></a>

Creates an object that either appends to an existing file, or writes to a new file (if the file does not exist to begin with).

On success, stores a pointer to the new file in \*result and returns OK. On failure stores NULL in \*result and returns non-OK.

The returned file will only be accessed by one thread at a time.

The ownership of the returned WritableFile is passed to the caller and the object should be deleted when is not used. The file object shouldn't live longer than the Env object.

#### `Status tensorflow::Env::NewReadOnlyMemoryRegionFromFile(const string &fname, ReadOnlyMemoryRegion **result)` <a href="#status_tensorflow_env_newreadonlymemoryregionfromfile" id="status_tensorflow_env_newreadonlymemoryregionfromfile"></a>

Creates a readonly region of memory with the file context.

On success, it returns a pointer to read-only memory region from the content of file fname. The ownership of the region is passed to the caller. On failure stores nullptr in \*result and returns non-OK.

The returned memory region can be accessed from many threads in parallel.

The ownership of the returned ReadOnlyMemoryRegion is passed to the caller and the object should be deleted when is not used. The memory region object shouldn't live longer than the Env object.

#### `bool tensorflow::Env::FileExists(const string &fname)` <a href="#bool_tensorflow_env_fileexists" id="bool_tensorflow_env_fileexists"></a>

Returns true iff the named file exists.

#### `Status tensorflow::Env::GetChildren(const string &dir, std::vector< string > *result)` <a href="#status_tensorflow_env_getchildren" id="status_tensorflow_env_getchildren"></a>

Stores in \*result the names of the children of the specified directory. The names are relative to "dir".

Original contents of \*results are dropped.

#### `Status tensorflow::Env::DeleteFile(const string &fname)` <a href="#status_tensorflow_env_deletefile" id="status_tensorflow_env_deletefile"></a>

Deletes the named file.

#### `Status tensorflow::Env::CreateDir(const string &dirname)` <a href="#status_tensorflow_env_createdir" id="status_tensorflow_env_createdir"></a>

Creates the specified directory.

#### `Status tensorflow::Env::DeleteDir(const string &dirname)` <a href="#status_tensorflow_env_deletedir" id="status_tensorflow_env_deletedir"></a>

Deletes the specified directory.

#### `Status tensorflow::Env::GetFileSize(const string &fname, uint64 *file_size)` <a href="#status_tensorflow_env_getfilesize" id="status_tensorflow_env_getfilesize"></a>

Stores the size of `fname` in `*file_size`.

#### `Status tensorflow::Env::RenameFile(const string &src, const string &target)` <a href="#status_tensorflow_env_renamefile" id="status_tensorflow_env_renamefile"></a>

Renames file src to target. If target already exists, it will be replaced.

#### `virtual uint64 tensorflow::Env::NowMicros()=0` <a href="#virtual_uint64_tensorflow_env_nowmicros" id="virtual_uint64_tensorflow_env_nowmicros"></a>

Returns the number of micro-seconds since some fixed point in time. Only useful for computing deltas of time.

#### `virtual void tensorflow::Env::SleepForMicroseconds(int micros)=0` <a href="#virtual_void_tensorflow_env_sleepformicroseconds" id="virtual_void_tensorflow_env_sleepformicroseconds"></a>

Sleeps/delays the thread for the prescribed number of micro-seconds.

#### `virtual Thread* tensorflow::Env::StartThread(const ThreadOptions &thread_options, const string &name, std::function< void()> fn) TF_MUST_USE_RESULT=0` <a href="#virtual_thread_tensorflow_env_startthread" id="virtual_thread_tensorflow_env_startthread"></a>

Returns a new thread that is running fn() and is identified (for debugging/performance-analysis) by "name".

Caller takes ownership of the result and must delete it eventually (the deletion will block until fn() stops running).

#### `virtual void tensorflow::Env::SchedClosure(std::function< void()> closure)=0` <a href="#virtual_void_tensorflow_env_schedclosure" id="virtual_void_tensorflow_env_schedclosure"></a>

#### `virtual void tensorflow::Env::SchedClosureAfter(int micros, std::function< void()> closure)=0` <a href="#virtual_void_tensorflow_env_schedclosureafter" id="virtual_void_tensorflow_env_schedclosureafter"></a>

#### `virtual Status tensorflow::Env::LoadLibrary(const char *library_filename, void **handle)=0` <a href="#virtual_status_tensorflow_env_loadlibrary" id="virtual_status_tensorflow_env_loadlibrary"></a>

#### `virtual Status tensorflow::Env::GetSymbolFromLibrary(void *handle, const char *symbol_name, void **symbol)=0` <a href="#virtual_status_tensorflow_env_getsymbolfromlibrary" id="virtual_status_tensorflow_env_getsymbolfromlibrary"></a>

#### `static Env* tensorflow::Env::Default()` <a href="#static_env_tensorflow_env_default" id="static_env_tensorflow_env_default"></a>

Returns a default environment suitable for the current operating system.

Sophisticated users may wish to provide their own Env implementation instead of relying on this default environment.

The result of Default() belongs to this library and must never be deleted.
