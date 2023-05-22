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


OrtStatus *IsCoreMLAvailable() {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  return status;
  // uint32_t coreml_flags = 0;
  // status = ort_api->SessionOptionsAppendExecutionProvider_CoreML(options, &coreml_flags);
  // return status;
}

OrtStatus *IsCUDAAvailable(int cuda_device_id) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = cuda_device_id;
  cuda_options.arena_extend_strategy = 1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  // cuda_options.gpu_mem_limit = 2L * 1024 * 1024 * 1024; // 2GB
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.do_copy_in_default_stream = 1;
  // cuda_options.cudnn_conv_use_max_workspace = 1;
  // cuda_options.cudnn_conv1d_pad_to_nc1d = 1;
  cuda_options.user_compute_stream = NULL;
  cuda_options.default_memory_arena_cfg = NULL;
  // status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_device_id);
  status = ort_api->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
  return status;
}

OrtSessionOptions *CreateSessionOptions() {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  return options;
}
OrtStatus *CreateSessionPathWithOptions(char *model_path,
  OrtEnv *env, OrtSession **out, OrtSessionOptions *options) {
  OrtStatus *status = NULL;
  if (status) return status;
  status = ort_api->CreateSession(env, model_path,
    options, out);
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

IONames GetIONames(const OrtSession* session) {
    IONames result;
    result.input_names = NULL;
    result.output_names = NULL;
    result.input_count = 0;
    result.output_count = 0;

    OrtStatus* status;
    OrtMemoryInfo* memory_info;
    status = ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        const char* error_message = ort_api->GetErrorMessage(status);
        printf("Failed to create memory info: %s\n", error_message);
        ort_api->ReleaseStatus(status);
        return result;
    }
    
    OrtAllocator* allocator;
    status = ort_api->CreateAllocator(session, memory_info, &allocator);
    if (status != NULL) {
        const char* error_message = ort_api->GetErrorMessage(status);
        printf("Failed to create allocator: %s\n", error_message);
        ort_api->ReleaseStatus(status);
        return result;
    }

    size_t input_count;
    status = ort_api->SessionGetInputCount(session, &input_count);
    if (status != NULL) {
        const char* error_message = ort_api->GetErrorMessage(status);
        printf("Failed to get input count: %s\n", error_message);
        ort_api->ReleaseStatus(status);
        return result;
    }

    // printf("Input count: %zu\n", input_count);

    char** input_names = (char**)malloc(input_count * sizeof(char*));
    for (int i = 0; i < input_count; i++) {
        char* input_name;
        status = ort_api->SessionGetInputName(session, i, allocator, &input_name);
        if (status != NULL) {
            const char* error_message = ort_api->GetErrorMessage(status);
            printf("Failed to get input name at index %d: %s\n", i, error_message);
            ort_api->ReleaseStatus(status);
            free(input_names);  // Free the previously allocated names
            return result;
        }

        // copy input_name to input_names[i]
        input_names[i] = (char*)malloc(strlen(input_name) + 1);
        strcpy(input_names[i], input_name);
        // printf("Input %d : name=%s\n", i, input_names[i]);
        // printf("Input %d : name=%s\n", i, input_name);

        status = ort_api->AllocatorFree(allocator, input_name);
        if (status != NULL) {
            const char* error_message = ort_api->GetErrorMessage(status);
            printf("Failed to free input name at index %d: %s\n", i, error_message);
            ort_api->ReleaseStatus(status);
            free(input_names);  // Free the previously allocated names
            return result;
        }
    }

    size_t output_count;
    status = ort_api->SessionGetOutputCount(session, &output_count);
    if (status != NULL) {
        const char* error_message = ort_api->GetErrorMessage(status);
        printf("Failed to get output count: %s\n", error_message);
        ort_api->ReleaseStatus(status);
        return result;
    }

    // printf("Output count: %zu\n", output_count);


    char** output_names = (char**)malloc(output_count * sizeof(char*));
    for (int i = 0; i < output_count; i++) {
        char* output_name;
        status = ort_api->SessionGetOutputName(session, i, allocator, &output_name);
        if (status != NULL) {
            const char* error_message = ort_api->GetErrorMessage(status);
            printf("Failed to get output name at index %d: %s\n", i, error_message);
            ort_api->ReleaseStatus(status);
            free(output_names);  // Free the previously allocated names
            return result;
        }

        // copy input_name to input_names[i]
        output_names[i] = (char*)malloc(strlen(output_name) + 1);
        strcpy(output_names[i], output_name);
        // printf("Output %d : name=%s\n", i, output_names[i]);
        // printf("Output %d : name=%s\n", i, output_name);

        status = ort_api->AllocatorFree(allocator, output_name);
        if (status != NULL) {
            const char* error_message = ort_api->GetErrorMessage(status);
            printf("Failed to free output name at index %d: %s\n", i, error_message);
            ort_api->ReleaseStatus(status);
            free(output_names);  // Free the previously allocated names
            return result;
        }
    }

    ort_api->ReleaseMemoryInfo(memory_info);
    ort_api->ReleaseAllocator(allocator);

    result.input_names = input_names;
    result.output_names = output_names;
    result.input_count = input_count;
    result.output_count = output_count;
    return result;
}

void FreeNames(char** names, int count) {
    if (names != NULL) {
        for (int i = 0; i < count; i++) {
            free(names[i]);
        }
        free(names);
    }
}

// OrtStatus *CreateSessionPathWithCoreML(char *model_path,
//   OrtEnv *env, OrtSession **out) {
//   OrtStatus *status = NULL;
//   OrtSessionOptions *options = NULL;
//   status = ort_api->CreateSessionOptions(&options);
//   if (status) return status;
//   uint32_t coreml_flags = 0;

//   // status = OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags);
//   // status = ort_api->SessionOptionsAppendExecutionProvider_CoreML(options, &coreml_flags);
//   if (status) return status;
//   status = ort_api->CreateSession(env, model_path,
//     options, out);
//   // It's OK to release the session options now, right? The docs don't say.
//   ort_api->ReleaseSessionOptions(options);
//   return status;
// }
OrtStatus *AppendExecutionProvider_CUDA(OrtSessionOptions *options, int cuda_device_id) {
  OrtStatus *status = NULL;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = cuda_device_id;
  cuda_options.arena_extend_strategy = 1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  // cuda_options.gpu_mem_limit = 2L * 1024 * 1024 * 1024; // 2GB
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.do_copy_in_default_stream = 1;
  // cuda_options.cudnn_conv_use_max_workspace = 1;
  // cuda_options.cudnn_conv1d_pad_to_nc1d = 1;
  cuda_options.user_compute_stream = NULL;
  cuda_options.default_memory_arena_cfg = NULL;
  // status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_device_id);
  status = ort_api->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
  return status;
}

OrtStatus *AppendExecutionProvider_TensorRT(OrtSessionOptions *options, int cuda_device_id, int trt_fp16_enable, int trt_int8_enable) {
  OrtStatus *status = NULL;
  OrtTensorRTProviderOptions cuda_options;
  cuda_options.device_id = cuda_device_id;
  cuda_options.trt_fp16_enable = trt_fp16_enable;
  cuda_options.trt_int8_enable = trt_int8_enable;
  // cuda_options.arena_extend_strategy = 1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  // cuda_options.gpu_mem_limit = 2L * 1024 * 1024 * 1024; // 2GB
  // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  // cuda_options.do_copy_in_default_stream = 1;
  // cuda_options.cudnn_conv_use_max_workspace = 1;
  // cuda_options.cudnn_conv1d_pad_to_nc1d = 1;
  // cuda_options.user_compute_stream = NULL;
  // cuda_options.default_memory_arena_cfg = NULL;
  // status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_device_id);
  status = ort_api->SessionOptionsAppendExecutionProvider_TensorRT(options, &cuda_options);
  return status;
}

OrtStatus *CreateSessionPathWithCUDA(char *model_path,
  OrtEnv *env, OrtSession **out, int cuda_device_id) {
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status) return status;
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = cuda_device_id;
  cuda_options.arena_extend_strategy = 1; // use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
  // cuda_options.gpu_mem_limit = 2L * 1024 * 1024 * 1024; // 2GB
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_options.do_copy_in_default_stream = 1;
  // cuda_options.cudnn_conv_use_max_workspace = 1;
  // cuda_options.cudnn_conv1d_pad_to_nc1d = 1;
  cuda_options.user_compute_stream = NULL;
  cuda_options.default_memory_arena_cfg = NULL;
  // status = OrtSessionOptionsAppendExecutionProvider_CUDA(options, cuda_device_id);
  status = ort_api->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
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
