#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;

int SetAPIFromBase(OrtApiBase *api_base)
{
  if (!api_base)
    return 1;
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api)
    return 2;
  return 0;
}

void ReleaseOrtStatus(OrtStatus *status)
{
  ort_api->ReleaseStatus(status);
}

OrtStatus *CreateOrtEnv(char *name, OrtEnv **env)
{
  return ort_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, env);
}

void ReleaseOrtEnv(OrtEnv *env)
{
  ort_api->ReleaseEnv(env);
}

OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info)
{
  return ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                      mem_info);
}

void ReleaseOrtMemoryInfo(OrtMemoryInfo *info)
{
  ort_api->ReleaseMemoryInfo(info);
}

const char *GetErrorMessage(OrtStatus *status)
{
  if (!status)
    return "No error (NULL status)";
  return ort_api->GetErrorMessage(status);
}

#ifdef _WIN32_SESSION
const wchar_t *GetWC(const char *c)
{
  const size_t cSize = strlen(c) + 1;
  wchar_t *wc = malloc(sizeof(wchar_t) * cSize);
  mbstowcs(wc, c, cSize);

  return wc;
}
#endif

OrtStatus *IsCoreMLAvailable()
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  return status;
  // uint32_t coreml_flags = 0;
  // status = ort_api->SessionOptionsAppendExecutionProvider_CoreML(options, &coreml_flags);
  // return status;
}

OrtStatus *IsCUDAAvailable(int cuda_device_id)
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status)
    return status;
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

OrtSessionOptions *CreateSessionOptions()
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  return options;
}
OrtStatus *CreateSessionPathWithOptions(char *model_path,
                                        OrtEnv *env, OrtSession **out, OrtSessionOptions *options)
{
  OrtStatus *status = NULL;
  if (status)
    return status;
#ifdef _WIN32_SESSION
  const wchar_t *model_path2 = GetWC(model_path);
  status = ort_api->CreateSession(env, (ORTCHAR_T *)model_path2,
                                  options, out);
  free((void *)model_path2);
#else
  status = ort_api->CreateSession(env, model_path,
                                  options, out);
#endif
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

//*****************************************************************************
// helper function to check for status
OrtStatus *CheckStatus(OrtStatus *status)
{
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Error message: %s\n", error_message);
    ort_api->ReleaseStatus(status);
  }
  return status;
}

IONames GetIONames(const OrtSession *session)
{
  IONames result;
  result.input_names = NULL;
  result.output_names = NULL;
  result.input_count = 0;
  result.output_count = 0;
  result.input_types = NULL;
  result.output_types = NULL;
  result.input_shapes = NULL;
  result.output_shapes = NULL;
  result.input_symbolic_shapes = NULL;
  result.output_symbolic_shapes = NULL;
  result.input_shapes_count = NULL;
  result.output_shapes_count = NULL;

  OrtStatus *status;
  OrtMemoryInfo *memory_info;
  status = ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to create memory info: %s\n", error_message);
    ort_api->ReleaseStatus(status);
    return result;
  }

  OrtAllocator *allocator;
  status = ort_api->CreateAllocator(session, memory_info, &allocator);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to create allocator: %s\n", error_message);
    ort_api->ReleaseStatus(status);
    return result;
  }

  size_t input_count;
  status = ort_api->SessionGetInputCount(session, &input_count);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input count: %s\n", error_message);
    ort_api->ReleaseStatus(status);
    return result;
  }

  // printf("Input count: %zu\n", input_count);

  char **input_names = (char **)malloc(input_count * sizeof(char *));

  char **input_types = (char **)malloc(input_count * sizeof(char *));
  int64_t **input_shapes = (int64_t **)malloc(input_count * sizeof(int64_t *));
  char ***input_symbolic_shapes = (char ***)malloc(input_count * sizeof(char **));
  int64_t *input_shapes_count = (int64_t *)malloc(input_count * sizeof(int64_t));

  // get input types
  // enum ONNXTensorElementDataType *input_types = (enum ONNXTensorElementDataType *)malloc(input_count * sizeof(enum ONNXTensorElementDataType));

  // get input dims
  // int64_t **input_shapes = (int64_t **)malloc(input_count * sizeof(int64_t *));

  for (int i = 0; i < input_count; i++)
  {
    char *input_name;
    status = ort_api->SessionGetInputName(session, i, allocator, &input_name);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input name at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }

    // copy input_name to input_names[i]
    input_names[i] = (char *)malloc(strlen(input_name) + 1);
    strcpy(input_names[i], input_name);
    // printf("Input %d : name=%s\n", i, input_names[i]);
    // printf("Input %d : name=%s\n", i, input_name);

    status = ort_api->AllocatorFree(allocator, input_name);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to free input name at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      // free input_names[i]
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }

    OrtTypeInfo *typeinfo;
    status = ort_api->SessionGetInputTypeInfo(session, i, &typeinfo);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input type info at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }
    const OrtTensorTypeAndShapeInfo *tensor_info;
    status = ort_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input tensor info at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }
    ONNXTensorElementDataType type;
    status = ort_api->GetTensorElementType(tensor_info, &type);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input tensor element type at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }
    // printf("Input %d : type=%d\n", i, type);

    input_types[i] = (char *)malloc(10 * sizeof(char));
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      strcpy(input_types[i], "float32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      strcpy(input_types[i], "uint8");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      strcpy(input_types[i], "int8");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      strcpy(input_types[i], "uint16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      strcpy(input_types[i], "int16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      strcpy(input_types[i], "int32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      strcpy(input_types[i], "int64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      strcpy(input_types[i], "string");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      strcpy(input_types[i], "bool");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      strcpy(input_types[i], "float16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      strcpy(input_types[i], "double");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      strcpy(input_types[i], "uint32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      strcpy(input_types[i], "uint64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      strcpy(input_types[i], "complex64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      strcpy(input_types[i], "complex128");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      strcpy(input_types[i], "bfloat16");
      break;
    default:
      strcpy(input_types[i], "unknown");
      break;
    }

    // print input shapes/dims
    size_t num_dims;
    status = ort_api->GetDimensionsCount(tensor_info, &num_dims);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input dimensions count at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(input_types[i]);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }
    // printf("Input %d : num_dims=%zu\n", i, num_dims);
    input_shapes_count[i] = num_dims;

    // get symbolic dimensions GetSymbolicDimensions
    const char **dim_params = (const char **)malloc(num_dims * sizeof(char *));
    status = ort_api->GetSymbolicDimensions(tensor_info, dim_params, num_dims);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input symbolic dimensions at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(dim_params);
      free(input_types[i]);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }
    // copy dim_params in to input_symbolic_shapes
    input_symbolic_shapes[i] = (char **)malloc(num_dims * sizeof(char *));
    for (size_t j = 0; j < num_dims; j++)
    {
      // printf("Input %d : dim %zu = %s\n", i, j, dim_params[j]);
      input_symbolic_shapes[i][j] = (char *)malloc(strlen(dim_params[j]) + 1);
      strcpy(input_symbolic_shapes[i][j], dim_params[j]);
    }
    // input_symbolic_shapes[i] = (char**)dim_params;

    int64_t *input_node_dims = (int64_t *)malloc(num_dims * sizeof(int64_t));
    status = ort_api->GetDimensions(tensor_info, input_node_dims, num_dims);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input dimensions at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(dim_params);
      free(input_node_dims);
      free(input_types[i]);
      free(input_names[i]);
      free(input_names); // Free the previously allocated names
      free(input_types);
      free(input_shapes);
      free(input_symbolic_shapes);
      free(input_shapes_count);
      return result;
    }

    input_shapes[i] = input_node_dims;

    // print dim_params
    // for (size_t j = 0; j < num_dims; j++)
    // {
    //   // check if strlen dim_params[j] == 0
    //   if (strlen(dim_params[j]) == 0)
    //   {
    //     printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
    //   }
    //   else
    //   {
    //     printf("Input %zu : dim %zu=%s\n", i, j, dim_params[j]);
    //   }
    // }

    // ort_api->ReleaseTensorTypeAndShapeInfo(tensor_info);
    ort_api->ReleaseTypeInfo(typeinfo);
  }

  size_t output_count;
  status = ort_api->SessionGetOutputCount(session, &output_count);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get output count: %s\n", error_message);
    ort_api->ReleaseStatus(status);
    return result;
  }

  // printf("Output count: %zu\n", output_count);

  char **output_names = (char **)malloc(output_count * sizeof(char *));

  char **output_types = (char **)malloc(output_count * sizeof(char *));
  char ***output_symbolic_shapes = (char ***)malloc(output_count * sizeof(char **));
  int64_t **output_shapes = (int64_t **)malloc(output_count * sizeof(int64_t *));
  int64_t *output_shapes_count = (int64_t *)malloc(output_count * sizeof(int64_t));

  for (int i = 0; i < output_count; i++)
  {
    char *output_name;
    status = ort_api->SessionGetOutputName(session, i, allocator, &output_name);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get output name at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    // copy input_name to input_names[i]
    output_names[i] = (char *)malloc(strlen(output_name) + 1);
    strcpy(output_names[i], output_name);
    // printf("Output %d : name=%s\n", i, output_names[i]);
    // printf("Output %d : name=%s\n", i, output_name);

    status = ort_api->AllocatorFree(allocator, output_name);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to free output name at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    // get output type info
    OrtTypeInfo *typeinfo;
    status = ort_api->SessionGetOutputTypeInfo(session, i, &typeinfo);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get output type info at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    // get output tensor info
    const OrtTensorTypeAndShapeInfo *tensor_info;
    status = ort_api->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to cast output type info to tensor info at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    // get output type
    ONNXTensorElementDataType output_type;
    status = ort_api->GetTensorElementType(tensor_info, &output_type);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get output type at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    // printf("Output %d : type=%d\n", i, output_type);
    output_types[i] = (char *)malloc(10);
    switch (output_type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      strcpy(output_types[i], "float32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      strcpy(output_types[i], "uint8");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      strcpy(output_types[i], "int8");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      strcpy(output_types[i], "uint16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      strcpy(output_types[i], "int16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      strcpy(output_types[i], "int32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      strcpy(output_types[i], "int64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      strcpy(output_types[i], "string");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      strcpy(output_types[i], "bool");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      strcpy(output_types[i], "float16");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      strcpy(output_types[i], "double");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      strcpy(output_types[i], "uint32");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      strcpy(output_types[i], "uint64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      strcpy(output_types[i], "complex64");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      strcpy(output_types[i], "complex128");
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      strcpy(output_types[i], "bfloat16");
      break;
    default:
      strcpy(output_types[i], "unknown");
      break;
    }

    // get output shape
    size_t output_dims_count;
    status = ort_api->GetDimensionsCount(tensor_info, &output_dims_count);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get output shape count at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(output_types[i]);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }
    output_shapes_count[i] = output_dims_count;

    // get symbolic dimensions GetSymbolicDimensions
    const char **dim_params = (const char **)malloc(output_dims_count * sizeof(char *));
    status = ort_api->GetSymbolicDimensions(tensor_info, dim_params, output_dims_count);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get input symbolic dimensions at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(dim_params);
      free(output_types[i]);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }
    // copy dim_params in to output_symbolic_shapes
    output_symbolic_shapes[i] = (char **)malloc(output_dims_count * sizeof(char *));
    for (int j = 0; j < output_dims_count; j++)
    {
      output_symbolic_shapes[i][j] = (char *)malloc(strlen(dim_params[j]) + 1);
      strcpy(output_symbolic_shapes[i][j], dim_params[j]);
    }

    // output_symbolic_shapes[i] = (char**)dim_params;

    int64_t *output_node_dims = (int64_t *)malloc(output_dims_count * sizeof(int64_t));
    status = ort_api->GetDimensions(tensor_info, output_node_dims, output_dims_count);
    if (status != NULL)
    {
      const char *error_message = ort_api->GetErrorMessage(status);
      printf("Failed to get output shape at index %d: %s\n", i, error_message);
      ort_api->ReleaseStatus(status);
      free(dim_params);
      free(output_node_dims);
      free(output_types[i]);
      free(output_names[i]);
      free(output_names); // Free the previously allocated names
      free(output_types);
      free(output_shapes);
      free(output_symbolic_shapes);
      free(output_shapes_count);
      return result;
    }

    output_shapes[i] = output_node_dims;

    // print dim_params
    // for (size_t j = 0; j < output_dims_count; j++)
    // {
    //   // check if strlen dim_params[j] == 0
    //   if (strlen(dim_params[j]) == 0)
    //   {
    //     printf("Output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);
    //   }
    //   else
    //   {
    //     printf("Output %zu : dim %zu=%s\n", i, j, dim_params[j]);
    //   }
    // }

    // ort_api->ReleaseTensorTypeAndShapeInfo(tensor_info);
    ort_api->ReleaseTypeInfo(typeinfo);
  }

  ort_api->ReleaseMemoryInfo(memory_info);
  ort_api->ReleaseAllocator(allocator);

  result.input_names = input_names;
  result.output_names = output_names;
  result.input_count = input_count;
  result.output_count = output_count;
  result.input_types = input_types;
  result.output_types = output_types;
  result.input_shapes = input_shapes;
  result.output_shapes = output_shapes;
  result.input_symbolic_shapes = input_symbolic_shapes;
  result.output_symbolic_shapes = output_symbolic_shapes;
  result.input_shapes_count = input_shapes_count;
  result.output_shapes_count = output_shapes_count;

  return result;
}

void FreeShapesCount(int64_t *counts)
{
  if (counts != NULL)
  {
    free(counts);
  }
}

void FreeTypes(char **types, int count)
{
  if (types != NULL)
  {
    for (int i = 0; i < count; i++)
    {
      free(types[i]);
    }
    free(types);
  }
}

void FreeShapes(int64_t **shapes, int count)
{
  if (shapes != NULL)
  {
    for (int i = 0; i < count; i++)
    {
      free(shapes[i]);
    }
    free(shapes);
  }
}

void FreeSymbolicShapes(char ***shapes, int64_t *counts, int count)
{
  if (shapes != NULL)
  {
    for (int i = 0; i < count; i++)
    {
      for (int j = 0; j < counts[i]; j++)
      {
        free(shapes[i][j]);
      }
      free(shapes[i]);
    }
    free(shapes);
  }
}

void FreeNames(char **names, int count)
{
  if (names != NULL)
  {
    for (int i = 0; i < count; i++)
    {
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
OrtStatus *AppendExecutionProvider_CUDA(OrtSessionOptions *options, int cuda_device_id)
{
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

OrtStatus *AppendExecutionProvider_TensorRT(OrtSessionOptions *options, int cuda_device_id, int trt_fp16_enable, int trt_int8_enable)
{
  OrtStatus *status = NULL;
  OrtTensorRTProviderOptions cuda_options;
  cuda_options.device_id = cuda_device_id;
  cuda_options.trt_max_workspace_size = 3 * 2147483648; // 6GB
  cuda_options.trt_fp16_enable = trt_fp16_enable;
  cuda_options.trt_int8_enable = trt_int8_enable;
  cuda_options.trt_engine_cache_enable = true;
  cuda_options.trt_engine_cache_path = ".trt_engine_cache";
  cuda_options.trt_engine_decryption_enable = false;
  status = ort_api->SessionOptionsAppendExecutionProvider_TensorRT(options, &cuda_options);
  return status;
}

OrtStatus *CreateSessionPathWithCUDA(char *model_path,
                                     OrtEnv *env, OrtSession **out, int cuda_device_id)
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status)
    return status;
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
  if (status)
    return status;
#ifdef _WIN32_SESSION
  const wchar_t *model_path2 = GetWC(model_path);
  status = ort_api->CreateSession(env, (ORTCHAR_T *)model_path2,
                                  options, out);
  free((void *)model_path2);
#else
  status = ort_api->CreateSession(env, model_path,
                                  options, out);
#endif
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *CreateSessionPath(char *model_path,
                             OrtEnv *env, OrtSession **out)
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status)
    return status;
#ifdef _WIN32_SESSION
  const wchar_t *model_path2 = GetWC(model_path);
  status = ort_api->CreateSession(env, (ORTCHAR_T *)model_path2,
                                  options, out);
  free((void *)model_path2);
#else
  status = ort_api->CreateSession(env, model_path,
                                  options, out);
#endif

  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *CreateSession(void *model_data, size_t model_data_length,
                         OrtEnv *env, OrtSession **out)
{
  OrtStatus *status = NULL;
  OrtSessionOptions *options = NULL;
  status = ort_api->CreateSessionOptions(&options);
  if (status)
    return status;
  status = ort_api->CreateSessionFromArray(env, model_data, model_data_length,
                                           options, out);
  // It's OK to release the session options now, right? The docs don't say.
  ort_api->ReleaseSessionOptions(options);
  return status;
}

OrtStatus *RunOrtSession(OrtSession *session,
                         OrtValue **inputs, char **input_names, int input_count,
                         OrtValue **outputs, char **output_names, int output_count)
{
  OrtStatus *status = NULL;
  status = ort_api->Run(session, NULL, (const char *const *)input_names,
                        (const OrtValue *const *)inputs, input_count,
                        (const char *const *)output_names, output_count, outputs);
  return status;
}

OrtStatus *RunOrtSessionIOCUDA(OrtSession *session,
                           OrtValue **inputs, char **input_names, int input_count,
                           OrtValue **outputs, char **output_names, int output_count, int cuda_device_id)
{
  OrtStatus *status = NULL;
  OrtIoBinding* io_binding;
  status = ort_api->CreateIoBinding(session, &io_binding);
  if (status) {
    return status;
    }

  for (int i = 0; i < input_count; i++)
  {
    status = ort_api->BindInput(io_binding, input_names[i], inputs[i]);
    if (status) {
      ort_api->ReleaseIoBinding(io_binding);
      return status;
    }
  }

  OrtMemoryInfo* memory_info;
  // status = ort_api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &memory_info);
  status = ort_api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, cuda_device_id, OrtMemTypeDefault, &memory_info);
  if (status) {
    ort_api->ReleaseIoBinding(io_binding);
    return status;
  }

  for (int i = 0; i < output_count; i++)
  {    
    status = ort_api->BindOutputToDevice(io_binding, output_names[i], memory_info);
    if (status) {
      ort_api->ReleaseIoBinding(io_binding);
      ort_api->ReleaseMemoryInfo(memory_info);
      return status;
    }
  }

  status = ort_api->RunWithBinding(session, NULL, io_binding);
  ort_api->ReleaseIoBinding(io_binding);
  ort_api->ReleaseMemoryInfo(memory_info);
  return status;
  // if (status) return status;
  // size_t numOutputs;
  // status = ort_api->SessionGetOutputCount(session, &numOutputs);
  // for (size_t i = 0; i < numOutputs; i++)
  // {
  //   OrtValue *output = outputs[i];
  //   const char *output_name = output_names[i];
  //   OrtTypeInfo *type_info;
  //   status = ort_api->SessionGetOutputTypeInfo(session, i, &type_info);
  //   if (status)
  //     return status;
  //   const OrtTensorTypeAndShapeInfo *outputTensorInfo;
  //   status = ort_api->CastTypeInfoToTensorInfo(type_info, &outputTensorInfo);
  //   if (status)
  //     return status;
  //   size_t num_dims;
  //   status = ort_api->GetDimensionsCount(outputTensorInfo, &num_dims);
  //   if (status)
  //     return status;
  //   int64_t *output_dims = malloc(num_dims * sizeof(int64_t));
  //   status = ort_api->GetDimensions(outputTensorInfo, output_dims, num_dims);
  //   if (status)
  //     return status;
  //   size_t output_size = 1;
  //   for (size_t j = 0; j < num_dims; j++)
  //   {
  //     output_size *= output_dims[j];
  //   }
  //   free(output_dims);
  //   float *output_data = malloc(output_size * sizeof(float));
  //   status = ort_api->GetTensorMutableData(output, (void **)&output_data);
  //   if (status)
  //     return status;
  //   // Do something with the output data here.
  //   free(output_data);
  // }
  // return status;
}

size_t GetTensorNumDimensions(OrtValue *value)
{
  OrtStatus *status = NULL;
  OrtTensorTypeAndShapeInfo *outputTensorInfo;
  status = ort_api->GetTensorTypeAndShape(value, &outputTensorInfo);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  size_t num_dims;
  status = ort_api->GetDimensionsCount(outputTensorInfo, &num_dims);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  ort_api->ReleaseTensorTypeAndShapeInfo(outputTensorInfo);
  return num_dims;
}

int64_t GetTensorDimensions(OrtValue *value, size_t j)
{
  OrtStatus *status = NULL;
  OrtTensorTypeAndShapeInfo *outputTensorInfo;
  status = ort_api->GetTensorTypeAndShape(value, &outputTensorInfo);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  size_t num_dims;
  status = ort_api->GetDimensionsCount(outputTensorInfo, &num_dims);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  int64_t *output_dims = malloc(num_dims * sizeof(int64_t));
  status = ort_api->GetDimensions(outputTensorInfo, output_dims, num_dims);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed: %s\n", error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  int64_t dim = output_dims[j];
  free(output_dims);
  ort_api->ReleaseTensorTypeAndShapeInfo(outputTensorInfo);
  return dim;
}

size_t GetTensorElementCount(OrtValue *value)
{
  OrtStatus *status = NULL;
  OrtTensorTypeAndShapeInfo *outputTensorInfo;
  status = ort_api->GetTensorTypeAndShape(value, &outputTensorInfo);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  size_t num_dims;
  status = ort_api->GetDimensionsCount(outputTensorInfo, &num_dims);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  int64_t *output_dims = malloc(num_dims * sizeof(int64_t));
  status = ort_api->GetDimensions(outputTensorInfo, output_dims, num_dims);
  if (status != NULL)
  {
    const char *error_message = ort_api->GetErrorMessage(status);
    printf("Failed to get input name at index %d: %s\n", (int)0, error_message);
    ort_api->ReleaseStatus(status);
    return 0;
  }
  size_t output_size = 1;
  for (size_t j = 0; j < num_dims; j++)
  {
    output_size *= output_dims[j];
  }
  free(output_dims);
  ort_api->ReleaseTensorTypeAndShapeInfo(outputTensorInfo);
  return output_size;
}

ONNXTensorElementDataType GetTensorElementType(OrtValue *output)
{
  OrtStatus *status = NULL;
  OrtTensorTypeAndShapeInfo *outputTensorInfo;
  status = ort_api->GetTensorTypeAndShape(output, &outputTensorInfo);
  if (status != NULL)
  {
    ort_api->ReleaseStatus(status);
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  ONNXTensorElementDataType elem_type;
  status = ort_api->GetTensorElementType(outputTensorInfo, &elem_type);
  if (status != NULL)
  {
    ort_api->ReleaseStatus(status);
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return elem_type;
}

OrtStatus *GetTensorMutableData(OrtValue *output, void *output_data)
{
  OrtStatus *status = NULL;
  ONNXTensorElementDataType elem_type = GetTensorElementType(output);
  void *tensor_data;
  status = ort_api->GetTensorMutableData(output, (void **)&tensor_data);

  size_t elementCount = GetTensorElementCount(output);

  switch (elem_type)
  {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
  {
    memcpy(output_data, tensor_data, sizeof(float) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
  {
    memcpy(output_data, tensor_data, sizeof(int32_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
  {
    memcpy(output_data, tensor_data, sizeof(int64_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  {
    memcpy(output_data, tensor_data, sizeof(double) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  {
    memcpy(output_data, tensor_data, sizeof(uint8_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  {
    memcpy(output_data, tensor_data, sizeof(uint16_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
  {
    memcpy(output_data, tensor_data, sizeof(uint32_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
  {
    memcpy(output_data, tensor_data, sizeof(uint64_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
  {
    memcpy(output_data, tensor_data, sizeof(bool) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
  {
    memcpy(output_data, tensor_data, sizeof(char) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
  {
    memcpy(output_data, tensor_data, sizeof(float) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
  {
    memcpy(output_data, tensor_data, sizeof(double) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
  {
    memcpy(output_data, tensor_data, sizeof(uint16_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  {
    memcpy(output_data, tensor_data, sizeof(uint16_t) * elementCount);
  }
  break;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
  {
    memcpy(output_data, tensor_data, sizeof(char) * elementCount);
  }
  break;

    // Handle other element types if needed

  default:
    // Unsupported data type
    break;
  }

  // copy tensor_data to output_data

  // printf("output_data: %f\n", ((float *)output_data)[0]);
  // printf("output_data: %f\n", ((float *)output_data)[1]);
  return status;
}

void ReleaseOrtSession(OrtSession *session)
{
  ort_api->ReleaseSession(session);
}

void ReleaseOrtValue(OrtValue *value)
{
  ort_api->ReleaseValue(value);
}

OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
                                    int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
                                    ONNXTensorElementDataType dtype, OrtValue **out)
{
  OrtStatus *status = NULL;
  status = ort_api->CreateTensorWithDataAsOrtValue(mem_info, data, data_size,
                                                   shape, shape_size, dtype, out);
  return status;
}
