/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/matmul_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

framework::DDim GetDimForInput(const framework::InferShapeContext &ctx,
                               std::string input_name) {
  auto shape = ctx.Attrs().Get<std::vector<int>>("fused_reshape_" + input_name);
  auto axis =
      ctx.Attrs().Get<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.GetInputDim(input_name);
  if (!shape.empty() && !axis.empty()) {
    PADDLE_ENFORCE_GE(
        shape.size(), 2,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_LE(
        shape.size(), 4,
        platform::errors::InvalidArgument(
            "shape_%s attribute of MatMulOp was implemented for 2, 3 "
            "or 4 dimensions.",
            input_name));
    PADDLE_ENFORCE_EQ(
        shape.size(), axis.size(),
        platform::errors::InvalidArgument(
            "Ranks of shape_%s and axis_%s attributes of MatMulOp "
            "must be equal.",
            input_name, input_name));
    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "matmul");

    auto dim_x = GetDimForInput(*context, "X");
    auto dim_y = GetDimForInput(*context, "Y");
    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0,
                                     context->Attrs().Get<bool>("transpose_X"));
    auto mat_dim_y =
        math::CreateMatrixDescriptor(ColumnMatrixFromVector(dim_y), 0,
                                     context->Attrs().Get<bool>("transpose_Y"));

    if (mat_dim_x.width_ == -1) {
      mat_dim_x.width_ = mat_dim_y.height_;
    }
    if (mat_dim_y.height_ == -1) {
      mat_dim_y.height_ = mat_dim_x.width_;
    }

    if (context->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
              mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0,
          true, platform::errors::InvalidArgument(
                    "The batch size of the two matrices should be equal, or "
                    "at least one is zero.\n"
                    "But received X's shape: %s, Y's shape: %s.",
                    DumpMatrixShape(mat_dim_x).c_str(),
                    DumpMatrixShape(mat_dim_y).c_str()));
    }
    int64_t dim_out_y = mat_dim_y.width_;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA)
    int head_number = context->Attrs().Get<int>("head_number");
    bool split_vertical_y = (mat_dim_x.width_ != mat_dim_y.height_);
    if (context->IsRuntime()) {
      PADDLE_ENFORCE_LE(
          head_number, mat_dim_x.width_,
          platform::errors::InvalidArgument(
              "Unsatisfied mkl acceleration library requirements: "
              "The number of heads "
              "(%d) must be equal to X's width. But received X's shape: %s.",
              head_number, DumpMatrixShape(mat_dim_x).c_str()));

      if (!split_vertical_y && head_number > 0) {
        dim_out_y = head_number * mat_dim_y.width_;
      }
    }
#else
    PADDLE_ENFORCE_EQ(mat_dim_x.width_, mat_dim_y.height_,
                      platform::errors::InvalidArgument(
                          "Input X's width should be equal to the Y's height, "
                          "but received X's shape: [%s],"
                          "Y's shape: [%s].",
                          dim_x, dim_y));
#endif

    std::vector<int64_t> dim_out;
    if (mat_dim_x.batch_size_ != 0) {
      dim_out = framework::vectorize(dim_x);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else if (mat_dim_y.batch_size_ != 0) {
      dim_out = framework::vectorize(dim_y);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else {
      dim_out = {mat_dim_x.height_, dim_out_y};
    }

    if (dim_x.size() == 1 && dim_out[dim_out.size() - 2] == 1) {
      std::swap(dim_out[dim_out.size() - 2], dim_out[dim_out.size() - 1]);
      dim_out.resize(dim_out.size() - 1);
    }

    if (dim_y.size() == 1 && dim_out[dim_out.size() - 1] == 1) {
      dim_out.resize(dim_out.size() - 1);
    }

    if (dim_out.empty()) {
      dim_out = {1};
    }

    framework::DDim ddim_out = framework::make_ddim(dim_out);

#ifdef PADDLE_WITH_MKLDNN
    //  if mkldnn matmul+transpose+reshape fuse activated
    auto reshape_out =
        context->Attrs().Get<std::vector<int>>("fused_reshape_Out");
    auto transpose_out =
        context->Attrs().Get<std::vector<int>>("fused_transpose_Out");

    if (!reshape_out.empty() && !transpose_out.empty()) {
      auto reshape_out_size = reshape_out.size();
      auto transpose_out_size = transpose_out.size();
      PADDLE_ENFORCE_EQ(transpose_out_size, 4,
                        platform::errors::InvalidArgument(
                            "transpose_out supported rank is 4, "
                            "received %d",
                            transpose_out_size));
      const std::vector<int> supported_axis{0, 2, 1, 3};
      const bool supported_transpose_axis = std::equal(
          transpose_out.begin(), transpose_out.end(), supported_axis.begin());
      PADDLE_ENFORCE_EQ(
          supported_transpose_axis, true,
          platform::errors::InvalidArgument(
              "supported transpose axis for the fuse are {0, 2, 1, 3}"));
      PADDLE_ENFORCE_EQ(
          reshape_out_size, 3,
          platform::errors::InvalidArgument("reshape_out supported rank is 3, "
                                            "received %d",
                                            reshape_out_size));
      framework::DDim shape_out =
          ddim_out.transpose(transpose_out).reshape(reshape_out);
      context->SetOutputDim("Out", shape_out);
    } else {
      context->SetOutputDim("Out", ddim_out);
    }
#else
    context->SetOutputDim("Out", ddim_out);
#endif
    context->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");

#ifdef PADDLE_WITH_MKLDNN
    using mkldnn::memory;
    if (platform::CanMKLDNNBeUsed(ctx)) {
      return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class MatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of MatMul op");
    AddInput("Y", "The second input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<bool>("transpose_X",
                  R"DOC(If true, use the transpose of `X`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y",
                  R"DOC(If true, use the transpose of `Y`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false);
    AddAttr<std::vector<int>>("fused_reshape_X",
                              R"DOC(Shape of fused reshape of `X` input.)DOC")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              R"DOC(Shape of fused reshape of `Y` input.)DOC")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_X",
                              R"DOC(Axis of fused transpose of `X` input.)DOC")
        .SetDefault({});
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              R"DOC(Axis of fused transpose of `Y` input.)DOC")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_reshape_Out",
        R"DOC(When MKLDNN MatMul_transpose_reshape fuse activated, "
              "it's a shape atribute of fused reshape for `Out` output.)DOC")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "fused_transpose_Out",
        R"DOC(When MKLDNN MatMul_transpose_reshape fuse activated, "
              "it's a axis atribute of fused transpose for `Out` output.)DOC")
        .SetDefault({});
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    /* int8 parameters */
    AddAttr<float>("Scale_x",
                   "(float, default 1.0f), The quantize scale of X tensor")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_y",
                   "(float, default 1.0f), The quantize scale of Y tensor")
        .SetDefault(1.0f);
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA)
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
#endif
    AddComment(R"DOC(
MatMul Operator.


This operator is used to perform (batched) matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`.

If a transpose flag is specified, the last two dimensions of the
tensor are transposed. If the tensor is rank-1 of shape [D], then
for `X` it is treated as [1, D] in nontransposed form and as [D, 1]
in transposed form, whereas for `Y` it is the opposite: It is treated
as [D, 1] in nontransposed form and as [1, D] in transposed form.

Examples without transpose:
- X: [K], Y: [K] => Out: [1]
- X: [K], Y: [K, N] => Out: [N]
- X: [B, M, K], Y: [K] => Out: [B, M]
- X: [M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, ..., M, K], Y: [B, ..., K, N] => Out: [B, ..., M, N]

Example of matrix multiplication with head_number of H
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, H * N]

The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.
- We add `head_number` attribute, which is used to multiple two matrixes head
  by head, and eventually concatenates the output of several (head_number)
  small matrixes multiplication.

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.

)DOC");
  }
};

class MatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "matmul");
    auto x_dims = context->GetInputDim("X");
    auto y_dims = context->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class MatMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("matmul_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul, ops::MatMulOp, ops::MatMulOpMaker,
                  ops::MatMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(matmul_grad, ops::MatMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    matmul, ops::MatMulCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulCPUKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, double>);
