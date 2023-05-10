#ifndef ONNXRUNTIME_WRAPPER_H
#define ONNXRUNTIME_WRAPPER_H

// We want to always use the unix-like onnxruntime C APIs, even on Windows, so
// we need to undefine _WIN32 before including onnxruntime_c_api.h. However,
// this requires a careful song-and-dance.

// First, include these common headers, as they get transitively included by
// onnxruntime_c_api.h. We need to include them ourselves, first, so that the
// preprocessor will skip then while _WIN32 is undefined.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define _WIN32_SESSION
#endif

// Next, we actually include the header.
#undef _WIN32
#include "onnxruntime_c_api.h"
#include "coreml_provider_factory.h"
// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32_SESSION
const wchar_t *GetWC(const char *c);
#endif

// Takes a pointer to the api_base struct in order to obtain the OrtApi
// pointer. Intended to be called from Go. Returns nonzero on error.
int SetAPIFromBase(OrtApiBase *api_base);

// Wraps ort_api->ReleaseStatus(status)
void ReleaseOrtStatus(OrtStatus *status);

// Wraps calling ort_api->CreateEnv. Returns a non-NULL status on error.
OrtStatus *CreateOrtEnv(char *name, OrtEnv **env);

// Wraps ort_api->ReleaseEnv
void ReleaseOrtEnv(OrtEnv *env);

// Wraps ort_api->CreateCpuMemoryInfo with some basic, default settings.
OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info);

// Wraps ort_api->ReleaseMemoryInfo
void ReleaseOrtMemoryInfo(OrtMemoryInfo *info);

// Returns the message associated with the given ORT status.
const char *GetErrorMessage(OrtStatus *status);

// Creates an ORT session using the given model.
OrtStatus *CreateSessionPath(char *model_path,
  OrtEnv *env, OrtSession **out);

// Check if CoreML is available
OrtStatus *IsCoreMLAvailable();

// Check if CUDA is available for the given device ID
OrtStatus *IsCUDAAvailable(int cuda_device_id);

// Creates an ORT session using the given model.
OrtStatus *CreateSessionPathWithCoreML(char *model_path,
  OrtEnv *env, OrtSession **out);

// Creates an ORT session using the given model.
OrtStatus *CreateSessionPathWithCUDA(char *model_path,
  OrtEnv *env, OrtSession **out, int cuda_device_id);

// Creates an ORT session using the given model.
OrtStatus *CreateSession(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out);

// Runs an ORT session with the given input and output tensors, along with
// their names. In our use case, outputs must NOT be NULL.
OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count);

// Wraps ort_api->ReleaseSession
void ReleaseOrtSession(OrtSession *session);

// Used to free OrtValue instances, such as tensors.
void ReleaseOrtValue(OrtValue *value);

// Creates an OrtValue tensor with the given shape, and backed by the user-
// supplied data buffer.
OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // ONNXRUNTIME_WRAPPER_H
