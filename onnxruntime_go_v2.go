//go:build !darwin

// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
	"strconv"
	// "os"
	// "github.com/ebitengine/purego"
	"unsafe"
)

/*
#cgo CFLAGS: -O2 -g
#include "onnxruntime_wrapper.h"
*/
import "C"

func NewSessionV2(path string, opts ...string) (*SessionV2, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	var ortSession *C.OrtSession
	modelPath := C.CString(path)
	defer C.free(unsafe.Pointer(modelPath))

	options := C.CreateSessionOptions()

	if len(opts) > 0 {
		// get device type from opts[0]
		deviceType := opts[0]
		if deviceType == "cuda" {
			// convert opts[0] from string to int
			deviceInt, err := strconv.Atoi(opts[1])
			if err != nil {
				deviceInt = 0
			}
			cudaDeviceId := C.int(deviceInt)
			status := C.AppendExecutionProvider_CUDA(options, cudaDeviceId)
			if status != nil {
				return nil, fmt.Errorf("Error creating session: %w",
					statusToError(status))
			}
		} else if deviceType == "tensorrt" {
			deviceInt, err := strconv.Atoi(opts[1])
			if err != nil {
				deviceInt = 0
			}
			fp16Int, err := strconv.Atoi(opts[2])
			if err != nil {
				fp16Int = 0
			}
			int8Int, err := strconv.Atoi(opts[3])
			if err != nil {
				int8Int = 0
			}
			deviceId := C.int(deviceInt)
			fp16 := C.int(fp16Int)
			int8 := C.int(int8Int)
			status := C.AppendExecutionProvider_TensorRT(options, deviceId, fp16, int8)
			if status != nil {
				return nil, fmt.Errorf("Error creating session: %w",
					statusToError(status))
			}
		}
	}

	// fmt.Printf("ortAPIBase: %v\n", ortAPIBase)

	status := C.CreateSessionPathWithOptions(modelPath, ortEnv, &ortSession, options)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	cNames := C.GetIONames(ortSession)

	// cInputNames := convertNames(inputNames)
	// cOutputNames := convertNames(outputNames)
	// inputOrtTensors := convertTensors(inputs)
	// outputOrtTensors := convertTensors(outputs)

	return &SessionV2{
		ortSession:  ortSession,
		inputNames:  cNames.input_names,
		outputNames: cNames.output_names,
		inputCount:  cNames.input_count,
		outputCount: cNames.output_count,
		// inputs:      inputOrtTensors,
		// outputs:     outputOrtTensors,
	}, nil
}

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithPathWithTypeWithCoreML(path string, inputNames,
	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType, opts ...string) (*Session, error) {
	panic("unimplemented")
}
