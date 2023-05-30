//go:build mac

package onnxruntime_go

/*
#cgo CFLAGS: -O2 -g
#cgo LDFLAGS: -ldl

#include "onnxruntime_wrapper.h"
typedef void (*AppendOptionsFunction)(OrtSessionOptions *options, uint32_t flags);

// Since Go can't call C function pointers directly, we just use this helper
// when calling GetApiBase
void CallAppendOptionsFunction(void *fn, OrtSessionOptions *options) {
	((AppendOptionsFunction) fn)(options, 0);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithPathWithTypeWithCoreML(path string, inputNames,
	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType, opts ...string) (*Session, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}

	err := checkInputsOutputs(inputs, outputs, inputNames, outputNames)
	if err != nil {
		return nil, err
	}

	var ortSession *C.OrtSession
	modelPath := C.CString(path)
	defer C.free(unsafe.Pointer(modelPath))

	options := C.CreateSessionOptions()
	cFunctionName := C.CString("OrtSessionOptionsAppendExecutionProvider_CoreML")
	defer C.free(unsafe.Pointer(cFunctionName))
	appendOptionsProc := C.dlsym(libraryHandle, cFunctionName)
	C.CallAppendOptionsFunction(appendOptionsProc, options)
	// fmt.Printf("ortAPIBase: %v\n", ortAPIBase)

	status := C.CreateSessionPathWithOptions(modelPath, ortEnv, &ortSession, options)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	cInputNames := convertNames(inputNames)
	cOutputNames := convertNames(outputNames)
	inputOrtTensors := convertTensors(inputs)
	outputOrtTensors := convertTensors(outputs)

	return &Session{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}

func SetupCoreMLExecutionProvider(options *_Ctype_struct_OrtSessionOptions) {
	cFunctionName := C.CString("OrtSessionOptionsAppendExecutionProvider_CoreML")
	defer C.free(unsafe.Pointer(cFunctionName))
	appendOptionsProc := C.dlsym(libraryHandle, cFunctionName)
	C.CallAppendOptionsFunction(appendOptionsProc, options)
}
