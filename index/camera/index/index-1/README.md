# 측면 카메라

0.5 버전의 텐서플로우의 퍼블릭 C++ API는 오직 그래프를 실행하는 API만을 포함합니다. C++ 로 부터 그래프 실행을 하는 것은 다음과 같습니다.

1. [Python API](../../../../g3doc/api\_docs/python/)를 이용해서 산출 그래프를 빌드합니다.
2. 그래프를 파일에 쓰기위해 [`tf.train.write_graph()`](broken-reference)를 이용합니다.
3. C++ 세션 API를 이용해 그래프를 읽어옵니다. 예를 들면:

```c++
// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(string graph_file_name, Session** session) {
  GraphDef graph_def;
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
  TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
  TF_RETURN_IF_ERROR((*session)->Create(graph_def));
  return Status::OK();
}
```

1. `session->Run()`을 호출하여 그래프를 보여줍니다.

## Env

* [tensorflow::Env](broken-reference)
* [tensorflow::RandomAccessFile](broken-reference)
* [tensorflow::WritableFile](broken-reference)
* [tensorflow::EnvWrapper](broken-reference)

## Session

* [tensorflow::Session](broken-reference)
* [tensorflow::SessionOptions](broken-reference)

## Status

* [tensorflow::Status](broken-reference)
* [tensorflow::Status::State](broken-reference)

## Tensor

* [tensorflow::Tensor](broken-reference)
* [tensorflow::TensorShape](broken-reference)
* [tensorflow::TensorShapeDim](broken-reference)
* [tensorflow::TensorShapeUtils](broken-reference)
* [tensorflow::PartialTensorShape](broken-reference)
* [tensorflow::PartialTensorShapeUtils](broken-reference)
* [TF\_Buffer](../../../../g3doc/api\_docs/cc/StructTF\_Buffer.md)

## Thread

* [tensorflow::Thread](broken-reference)
* [tensorflow::ThreadOptions](broken-reference)
