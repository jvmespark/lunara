// third_party/onnx/onnx_pb_stub.h
// Minimal stub of ONNX protobuf types used by Lunara's frontend.
// This lets the project build on systems without libonnx installed —
// the importer will run but produce empty modules.  Install real libonnx
// (e.g. `apt install libonnx-dev` or `pip install onnx` and link
// against the C++ headers) for actual ONNX parsing.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace onnx {

struct TensorShapeProto {
  struct Dimension {
    int64_t     dim_value() const { return value; }
    std::string dim_param;
    int64_t     value = 0;
  };
  std::vector<Dimension> dim_;
  int dim_size() const { return static_cast<int>(dim_.size()); }
  const Dimension &dim(int i) const { return dim_[i]; }
};

struct TypeProto {
  enum class ValueCase { kTensorType, kSequenceType, kOptionalType };
  struct Tensor {
    int              elem_type = 1;
    TensorShapeProto shape;
  };
  Tensor tensor_type_;
  const Tensor &tensor_type() const { return tensor_type_; }
};

struct TensorProto {
  enum DataType : int {
    UNDEFINED = 0,
    FLOAT     = 1,
    UINT8     = 2,
    INT8      = 3,
    UINT16    = 4,
    INT16     = 5,
    INT32     = 6,
    INT64     = 7,
    STRING    = 8,
    BOOL      = 9,
    FLOAT16   = 10,
    DOUBLE    = 11,
    UINT32    = 12,
    UINT64    = 13,
    BFLOAT16  = 16,
  };
  std::string          name;
  std::vector<int64_t> dims_;
  int                  data_type = FLOAT;
  std::string          raw_data;
};

struct AttributeProto {
  std::string name;
  enum AttributeType { UNDEFINED = 0, FLOAT = 1, INT = 2, STRING = 3,
                       TENSOR = 4, FLOATS = 6, INTS = 7, STRINGS = 8 };
  AttributeType type = UNDEFINED;
  float        f      = 0.0f;
  int64_t      i      = 0;
  std::string  s;
  TensorProto  t;
  std::vector<float>   floats;
  std::vector<int64_t> ints;
};

struct NodeProto {
  std::string                  op_type;
  std::string                  name;
  std::vector<std::string>     input;
  std::vector<std::string>     output;
  std::vector<AttributeProto>  attribute_;
  int input_size()  const { return static_cast<int>(input.size()); }
  int output_size() const { return static_cast<int>(output.size()); }
  const std::string &input(int i)  const { return input[i]; }
  const std::string &output(int i) const { return output[i]; }
  const std::vector<AttributeProto> &attribute() const { return attribute_; }
};

struct ValueInfoProto {
  std::string name;
  TypeProto   type;
};

struct GraphProto {
  std::string                 name;
  std::vector<NodeProto>      node;
  std::vector<ValueInfoProto> input;
  std::vector<ValueInfoProto> output;
  std::vector<TensorProto>    initializer;
};

struct ModelProto {
  GraphProto graph_;
  int64_t    ir_version = 7;
  GraphProto graph() const { return graph_; }
  bool ParseFromString(const std::string &) { return false; }
};

} // namespace onnx
