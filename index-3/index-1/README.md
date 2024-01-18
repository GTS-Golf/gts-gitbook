# C++ API

0.5 버전의 텐서플로우의 퍼블릭 C++ API는 오직 그래프를 실행하는 API만을 포함합니다. C++ 로 부터 그래프 실행을 하는 것은 다음과 같습니다.

1. [Python API](../../g3doc/api\_docs/python/)를 이용해서 산출 그래프를 빌드합니다.
2. 그래프를 파일에 쓰기위해 [`tf.train.write_graph()`](../index/train.md#write\_graph)를 이용합니다.
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

* [tensorflow::Env](classenv.md)
* [tensorflow::RandomAccessFile](classrandomaccessfile.md)
* [tensorflow::WritableFile](classwritablefile.md)
* [tensorflow::EnvWrapper](classenvwrapper.md)

## Session

* [tensorflow::Session](classsession.md)
* [tensorflow::SessionOptions](structsessionoptions.md)

## Status

* [tensorflow::Status](classstatus.md)
* [tensorflow::Status::State](structstate.md)

## Tensor

* [tensorflow::Tensor](classtensor.md)
* [tensorflow::TensorShape](classtensorshape.md)
* [tensorflow::TensorShapeDim](structtensorshapedim.md)
* [tensorflow::TensorShapeUtils](classtensorshapeutils.md)
* [tensorflow::PartialTensorShape](classpartialtensorshape.md)
* [tensorflow::PartialTensorShapeUtils](classpartialtensorshapeutils.md)
* [TF\_Buffer](../../g3doc/api\_docs/cc/StructTF\_Buffer.md)

## Thread

* [tensorflow::Thread](classthread.md)
* [tensorflow::ThreadOptions](structthreadoptions.md)
