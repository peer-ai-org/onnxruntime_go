//go:build !darwin

// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
	// "math"
	"strconv"
	"unsafe"
	// "strconv"
	// "os"
	// "github.com/ebitengine/purego"
	// "unsafe"
)

/*
#cgo CFLAGS: -O2 -g
#include "onnxruntime_wrapper.h"
*/
import "C"

func NewSessionV3(path string, opts ...string) (*SessionV3, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	var ortSession *C.OrtSession
	modelPath := C.CString(path)
	defer C.free(unsafe.Pointer(modelPath))

	options := C.CreateSessionOptions()
	var deviceInt int
	var deviceType string
	var err error

	if len(opts) > 0 {
		// get device type from opts[0]
		deviceType = opts[0]
		if deviceType == "cuda" {
			// append tensorrt first
			deviceInt, err = strconv.Atoi(opts[1])
			if err != nil {
				deviceInt = 0
			}
			fp16Int, err := strconv.Atoi(opts[2])
			if err != nil {
				fp16Int = 1
			}
			int8Int, err := strconv.Atoi(opts[3])
			if err != nil {
				int8Int = 1
			}
			deviceId := C.int(deviceInt)
			fp16 := C.int(fp16Int)
			int8 := C.int(int8Int)
			status := C.AppendExecutionProvider_TensorRT(options, deviceId, fp16, int8)
			if status != nil {
				return nil, fmt.Errorf("error creating session: %w",
					statusToError(status))
			}

			// convert opts[0] from string to int
			deviceInt, err = strconv.Atoi(opts[1])
			if err != nil {
				deviceInt = 0
			}
			cudaDeviceId := C.int(deviceInt)
			status = C.AppendExecutionProvider_CUDA(options, cudaDeviceId)
			if status != nil {
				return nil, fmt.Errorf("error creating session: %w",
					statusToError(status))
			}
		} else if deviceType == "tensorrt" {
			deviceInt, err = strconv.Atoi(opts[1])
			if err != nil {
				deviceInt = 0
			}
			fp16Int, err := strconv.Atoi(opts[2])
			if err != nil {
				fp16Int = 1
			}
			int8Int, err := strconv.Atoi(opts[3])
			if err != nil {
				int8Int = 1
			}
			deviceId := C.int(deviceInt)
			fp16 := C.int(fp16Int)
			int8 := C.int(int8Int)
			status := C.AppendExecutionProvider_TensorRT(options, deviceId, fp16, int8)
			if status != nil {
				return nil, fmt.Errorf("error creating session: %w",
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

	// cInputNames := ConvertNames(inputNames)
	// cOutputNames := ConvertNames(outputNames)
	// inputOrtTensors := ConvertTensors(inputs)
	// outputOrtTensors := ConvertTensors(outputs)

	// convert cgo to c variables
	// char** input_types;
	// int** input_shapes;
	// char*** input_symbolic_shapes;

	return &SessionV3{
		ortSession:           ortSession,
		inputNames:           cNames.input_names,
		outputNames:          cNames.output_names,
		inputCount:           cNames.input_count,
		outputCount:          cNames.output_count,
		inputTypes:           cNames.input_types,
		inputShapes:          cNames.input_shapes,
		inputSymbolicShapes:  cNames.input_symbolic_shapes,
		inputShapesCount:     cNames.input_shapes_count,
		outputTypes:          cNames.output_types,
		outputShapes:         cNames.output_shapes,
		outputSymbolicShapes: cNames.output_symbolic_shapes,
		outputShapesCount:    cNames.output_shapes_count,

		deviceType: deviceType,
		deviceId:   deviceInt,

		// inputs:      inputOrtTensors,
		// outputs:     outputOrtTensors,
	}, nil
}
