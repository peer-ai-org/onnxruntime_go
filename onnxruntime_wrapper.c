#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;

int SetAPIFromBase(OrtApiBase *api_base) {
  if (!api_base) return 1;
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api) return 2;
  return 0;
}

void ReleaseOrtStatus(OrtStatus *status) {
  ort_api->ReleaseStatus(status);
}

OrtStatus *CreateOrtEnv(char *name, OrtEnv **env) {
  return ort_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, env);
}

void ReleaseOrtEnv(OrtEnv *env) {
  ort_api->ReleaseEnv(env);
}

OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info) {
  return ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
    mem_info);
}

void ReleaseOrtMemoryInfo(OrtMemoryInfo *info) {
  ort_api->ReleaseMemoryInfo(info);
}

const char *GetErrorMessage(OrtStatus *status) {
  if (!status) return "No error (NULL status)";
  return ort_api->GetErrorMessage(status);
}

#ifdef _WIN32_SESSION
const wchar_t *GetWC(const char *c)
{
    const size_t cSize = strlen(c)+1;
    wchar_t* wc = malloc(sizeof(wchar_t)*cSize);
    mbstowcs (wc, c, cSize);

    return wc;
}
#endif

OrtStatus *CreateSessionPathWithCUDA(char *model_path,
  OrtEnv *env, OrtSession **out) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  cuda_options.arena_extend_strategy = 0; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  // cuda_options.gpu_mem_limit = 2L * 1024 * 1024 * 1024;
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.do_copy_in_default_stream = 1;
  // cuda_options.cudnn_conv_use_max_workspace = 1;
  // cuda_options.cudnn_conv1d_pad_to_nc1d = 1;
  cuda_options.user_compute_stream = NULL;
  cuda_options.default_memory_arena_cfg = NULL;
  status = ort_api->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
  // if (status) return status;
  #ifdef _WIN32_SESSION
  const wchar_t* model_path2 = GetWC(model_path);
  status = ort_api->CreateSession(env, (ORTCHAR_T*)model_path2,
    options, out);
  free((void*)model_path2);
  #else
  status = ort_api->CreateSession(env, model_path,
    options, out);
  #endif
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *CreateSessionPath(char *model_path,
  OrtEnv *env, OrtSession **out) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  #ifdef _WIN32_SESSION
  const wchar_t* model_path2 = GetWC(model_path);
  status = ort_api->CreateSession(env, (ORTCHAR_T*)model_path2,
    options, out);
  free((void*)model_path2);
  #else
  status = ort_api->CreateSession(env, model_path,
    options, out);
  #endif

  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *CreateSession(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  status = ort_api->CreateSessionFromArray(env, model_data, model_data_length,
    options, out);
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count) {
  OrtStatus *status = NULL;
  status = ort_api->Run(session, NULL, (const char* const*) input_names,
    (const OrtValue* const*) inputs, input_count,
    (const char* const*) output_names, output_count, outputs);
  return status;
}

void ReleaseOrtSession(OrtSession *session) {
  ort_api->ReleaseSession(session);
}

void ReleaseOrtValue(OrtValue *value) {
  ort_api->ReleaseValue(value);
}

OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out) {
  OrtStatus *status = NULL;
  status = ort_api->CreateTensorWithDataAsOrtValue(mem_info, data, data_size,
    shape, shape_size, dtype, out);
  return status;
}
