// Data-movement kernels with the copy plan worked out once.
//
// The streaming wake word model updates its internal state with 10 STRIDED_SLICE, 8 CONCATENATION and 2 SPLIT_V per
// inference. They only copy bytes, but TFLite Micro's reference kernels re-derive the geometry on every call (together
// ~24 % of the model's time on the ESP32-S3). Shapes and parameters are constant, so Prepare() decides once whether
// the operation is a few contiguous copies; Eval() is then memcpy. Anything else falls back to the reference code.
// Pure data movement: the outputs are bit-identical (checked on 226 clips, benchmarks/results/F4_*).
#include <cstring>

#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/strided_slice.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace {

struct SliceData {
    tflite::StridedSliceParams params;  // first member: StridedSlicePrepare() fills it through node->user_data
    int32_t offset_bytes;
    int32_t length_bytes;               // < 0: not one contiguous block -> reference kernel
};

void *SliceInit(TfLiteContext *context, const char *, size_t) {
    auto *d = static_cast<SliceData *>(context->AllocatePersistentBuffer(context, sizeof(SliceData)));
    if (d) d->length_bytes = -1;
    return d;
}

TfLiteStatus SlicePrepare(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE_STATUS(tflite::StridedSlicePrepare(context, node));
    auto *d = static_cast<SliceData *>(node->user_data);
    d->length_bytes = -1;

    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *input = mc->AllocateTempInputTensor(node, tflite::kStridedSliceInputTensor);
    const tflite::RuntimeShape shape = tflite::RuntimeShape::ExtendedShape(5, tflite::GetTensorShape(input));
    const size_t elem = input->type == kTfLiteInt8 ? 1 : input->type == kTfLiteInt16 ? 2
                        : (input->type == kTfLiteInt32 || input->type == kTfLiteFloat32) ? 4 : 0;
    mc->DeallocateTempTfLiteTensor(input);
    if (!elem) return kTfLiteOk;

    // Same index logic as reference_ops::StridedSlice (5-D, padded)
    tflite::StridedSliceParams p = d->params;
    tflite::strided_slice::StridedSlicePadIndices(&p, 5);
    int start[5], ext[5];
    for (int a = 0; a < 5; a++) {
        if (p.strides[a] != 1) return kTfLiteOk;
        start[a] = tflite::strided_slice::StridedSliceStartForAxis(p, shape, a);
        const int stop = tflite::strided_slice::StridedSliceEndForAxis(p, shape, a, start[a]);
        ext[a] = stop - start[a];
        if (ext[a] <= 0) return kTfLiteOk;
    }
    // One block: from the innermost axis outwards, full axes, then at most one partial axis, then extent-1 axes.
    int a = 4;
    while (a > 0 && start[a] == 0 && ext[a] == shape.Dims(a)) a--;
    for (int o = a - 1; o >= 0; o--)
        if (ext[o] != 1) return kTfLiteOk;
    int32_t offset = 0, stride = 1, length = 1;
    for (int k = 4; k >= 0; k--) {
        offset += start[k] * stride;
        length *= ext[k];
        stride *= shape.Dims(k);
    }
    d->offset_bytes = offset * (int32_t)elem;
    d->length_bytes = length * (int32_t)elem;
    return kTfLiteOk;
}

TfLiteStatus SliceEval(TfLiteContext *context, TfLiteNode *node) {
    const auto *d = static_cast<const SliceData *>(node->user_data);
    const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, tflite::kStridedSliceInputTensor);
    TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, tflite::kStridedSliceOutputTensor);
    if (d->length_bytes >= 0) {
        std::memcpy(output->data.raw, static_cast<const char *>(input->data.raw) + d->offset_bytes, d->length_bytes);
        return kTfLiteOk;
    }
    switch (output->type) {
        case kTfLiteInt8:
            tflite::reference_ops::StridedSlice(d->params, tflite::micro::GetTensorShape(input),
                                                tflite::micro::GetTensorData<int8_t>(input),
                                                tflite::micro::GetTensorShape(output),
                                                tflite::micro::GetTensorData<int8_t>(output));
            return kTfLiteOk;
        case kTfLiteFloat32:
            tflite::reference_ops::StridedSlice(d->params, tflite::micro::GetTensorShape(input),
                                                tflite::micro::GetTensorData<float>(input),
                                                tflite::micro::GetTensorShape(output),
                                                tflite::micro::GetTensorData<float>(output));
            return kTfLiteOk;
        default:
            return kTfLiteError;
    }
}

// ---- CONCATENATION: TFLite Micro's kernel (int8 is a plain copy) rebuilds shape arrays on every call. When nothing
// comes before the concatenation axis (outer size 1, the model's state buffers), the output is the inputs back to back.
struct ConcatData {
    void *ref;              // the reference kernel's own OpData (its Prepare/Eval use it through node->user_data)
    int n;                  // inputs, 0 = use the reference kernel
    int32_t bytes[10];
};
const TFLMRegistration &concat_ref() {
    static const TFLMRegistration r = tflite::Register_CONCATENATION();
    return r;
}

void *ConcatInit(TfLiteContext *context, const char *buffer, size_t length) {
    auto *d = static_cast<ConcatData *>(context->AllocatePersistentBuffer(context, sizeof(ConcatData)));
    if (!d) return nullptr;
    d->ref = concat_ref().init(context, buffer, length);
    d->n = 0;
    return d;
}

TfLiteStatus ConcatPrepare(TfLiteContext *context, TfLiteNode *node) {
    auto *d = static_cast<ConcatData *>(node->user_data);
    node->user_data = d->ref;
    const TfLiteStatus st = concat_ref().prepare(context, node);
    node->user_data = d;
    TF_LITE_ENSURE_STATUS(st);
    d->n = 0;
    const int n = node->inputs->size;
    if (n > 10) return kTfLiteOk;
    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *out = mc->AllocateTempOutputTensor(node, 0);
    int axis = reinterpret_cast<TfLiteConcatenationParams *>(node->builtin_data)->axis;
    if (axis < 0) axis += out->dims->size;
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= out->dims->data[i];
    mc->DeallocateTempTfLiteTensor(out);
    if (outer != 1) return kTfLiteOk;
    for (int i = 0; i < n; i++) {
        TfLiteTensor *in = mc->AllocateTempInputTensor(node, i);
        d->bytes[i] = (int32_t)in->bytes;
        mc->DeallocateTempTfLiteTensor(in);
    }
    d->n = n;
    return kTfLiteOk;
}

TfLiteStatus ConcatEval(TfLiteContext *context, TfLiteNode *node) {
    auto *d = static_cast<ConcatData *>(node->user_data);
    if (d->n == 0) {
        node->user_data = d->ref;
        const TfLiteStatus st = concat_ref().invoke(context, node);
        node->user_data = d;
        return st;
    }
    char *dst = static_cast<char *>(tflite::micro::GetEvalOutput(context, node, 0)->data.raw);
    for (int i = 0; i < d->n; i++) {
        std::memcpy(dst, tflite::micro::GetEvalInput(context, node, i)->data.raw, d->bytes[i]);
        dst += d->bytes[i];
    }
    return kTfLiteOk;
}

// ---- SPLIT_V: the reference copies element by element; the same copies with memcpy.
TfLiteStatus SplitPrepare(TfLiteContext *context, TfLiteNode *node) {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *axis = mc->AllocateTempInputTensor(node, 2);
    TF_LITE_ENSURE_MSG(context, tflite::IsConstantTensor(axis), "Non-constant >axis< tensor is not supported");
    mc->DeallocateTempTfLiteTensor(axis);
    return kTfLiteOk;
}

TfLiteStatus SplitEval(TfLiteContext *context, TfLiteNode *node) {
    const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
    int axis = tflite::micro::GetTensorData<int32_t>(tflite::micro::GetEvalInput(context, node, 2))[0];
    if (axis < 0) axis += input->dims->size;
    TF_LITE_ENSURE(context, axis >= 0 && axis < input->dims->size);
    size_t elem = 0;
    TF_LITE_ENSURE_STATUS(tflite::TfLiteTypeSizeOf(input->type, &elem));
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= input->dims->data[i];
    for (int i = axis + 1; i < input->dims->size; i++) inner *= input->dims->data[i];
    const int outputs = node->outputs->size;
    const char *src = static_cast<const char *>(input->data.raw);
    for (int64_t k = 0; k < outer; k++) {
        for (int i = 0; i < outputs; i++) {
            TfLiteEvalTensor *o = tflite::micro::GetEvalOutput(context, node, i);
            const size_t len = (size_t)(o->dims->data[axis] * inner) * elem;
            std::memcpy(static_cast<char *>(o->data.raw) + k * len, src, len);
            src += len;
        }
    }
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_STRIDED_SLICE_FAST() { return tflite::micro::RegisterOp(SliceInit, SlicePrepare, SliceEval); }
TFLMRegistration Register_CONCATENATION_FAST() { return tflite::micro::RegisterOp(ConcatInit, ConcatPrepare, ConcatEval); }
TFLMRegistration Register_SPLIT_V_FAST() { return tflite::micro::RegisterOp(nullptr, SplitPrepare, SplitEval); }
