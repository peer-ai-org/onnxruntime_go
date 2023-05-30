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
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include "onnxruntime_wrapper.h"
typedef void (*AppendOptionsFunction)(OrtSessionOptions *options, uint32_t flags);

// Since Go can't call C function pointers directly, we just use this helper
// when calling GetApiBase
void CallAppendOptionsFunction(void *fn, OrtSessionOptions *options) {
	((AppendOptionsFunction) fn)(options, 0);
}
*/
import "C"

// This string should be the path to onnxruntime.so, or onnxruntime.dll.
var onnxSharedLibraryPath string

// For simplicity, this library maintains a single ORT environment internally.
var ortEnv *C.OrtEnv

// We also keep a single OrtMemoryInfo value around, since we only support CPU
// allocations for now.
var ortMemoryInfo *C.OrtMemoryInfo

var NotInitializedError error = fmt.Errorf("InitializeRuntime() has either " +
	"not yet been called, or did not return successfully")

// Does two things: converts the given OrtStatus to a Go error, and releases
// the status. If the status is nil, this does nothing and returns nil.
func statusToError(status *C.OrtStatus) error {
	if status == nil {
		return nil
	}
	msg := C.GetErrorMessage(status)
	toReturn := C.GoString(msg)
	C.ReleaseOrtStatus(status)
	return fmt.Errorf("%s", toReturn)
}

// Use this function to set the path to the "onnxruntime.so" or
// "onnxruntime.dll" function. By default, it will be set to "onnxruntime.so"
// on non-Windows systems, and "onnxruntime.dll" on Windows. Users wishing to
// specify a particular location of this library must call this function prior
// to calling onnxruntime.InitializeEnvironment().
func SetSharedLibraryPath(path string) {
	onnxSharedLibraryPath = path
}

// Returns false if the onnxruntime package is not initialized. Called
// internally by several functions, to avoid segfaulting if
// InitializeEnvironment hasn't been called yet.
func IsInitialized() bool {
	return ortEnv != nil
}

// Call this function to initialize the internal onnxruntime environment. If
// this doesn't return an error, the caller will be responsible for calling
// DestroyEnvironment to free the onnxruntime state when no longer needed.
func InitializeEnvironment() error {
	if IsInitialized() {
		return fmt.Errorf("The onnxruntime has already been initialized")
	}
	// Do the windows- or linux- specific initialization first.
	e := platformInitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Platform-specific initialization failed: %w", e)
	}

	name := C.CString("Golang onnxruntime environment")
	defer C.free(unsafe.Pointer(name))
	status := C.CreateOrtEnv(name, &ortEnv)
	if status != nil {
		return fmt.Errorf("Error creating ORT environment: %w",
			statusToError(status))
	}

	status = C.CreateOrtMemoryInfo(&ortMemoryInfo)
	if status != nil {
		DestroyEnvironment()
		return fmt.Errorf("Error creating ORT memory info: %w",
			statusToError(status))
	}

	return nil
}

// Call this function to cleanup the internal onnxruntime environment when it
// is no longer needed.
func DestroyEnvironment() error {
	var e error
	if !IsInitialized() {
		return NotInitializedError
	}
	if ortMemoryInfo != nil {
		C.ReleaseOrtMemoryInfo(ortMemoryInfo)
		ortMemoryInfo = nil
	}
	if ortEnv != nil {
		C.ReleaseOrtEnv(ortEnv)
		ortEnv = nil
	}

	// platformCleanup primarily unloads the library, so we need to call it
	// last, after any functions that make use of the ORT API.
	e = platformCleanup()
	if e != nil {
		return fmt.Errorf("Platform-specific cleanup failed: %w", e)
	}
	return nil
}

// The Shape type holds the shape of the tensors used by the network input and
// outputs.
type Shape []int64

// Returns a Shape, with the given dimensions.
func NewShape(dimensions ...int64) Shape {
	return Shape(dimensions)
}

// Returns the total number of elements in a tensor with the given shape.
func (s Shape) FlattenedSize() int64 {
	if len(s) == 0 {
		return 0
	}
	toReturn := int64(s[0])
	for i := 1; i < len(s); i++ {
		toReturn *= s[i]
	}
	return toReturn
}

// Makes and returns a deep copy of the Shape.
func (s Shape) Clone() Shape {
	toReturn := make([]int64, len(s))
	copy(toReturn, []int64(s))
	return Shape(toReturn)
}

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

type TensorWithType struct {
	TensorType string
	Tensor     interface{}
}

func (t *TensorWithType) GetShape() []int64 {
	switch t.TensorType {
	case "float32":
		return t.Tensor.(*Tensor[float32]).GetShape()
	case "float64":
		return t.Tensor.(*Tensor[float64]).GetShape()
	case "int8":
		return t.Tensor.(*Tensor[int8]).GetShape()
	case "int16":
		return t.Tensor.(*Tensor[int16]).GetShape()
	case "int32":
		return t.Tensor.(*Tensor[int32]).GetShape()
	case "int64":
		return t.Tensor.(*Tensor[int64]).GetShape()
	case "uint8":
		return t.Tensor.(*Tensor[uint8]).GetShape()
	case "uint16":
		return t.Tensor.(*Tensor[uint16]).GetShape()
	case "uint32":
		return t.Tensor.(*Tensor[uint32]).GetShape()
	case "uint64":
		return t.Tensor.(*Tensor[uint64]).GetShape()
	case "bool":
		return t.Tensor.(*Tensor[bool]).GetShape()
	}
	return nil
}

func (t *TensorWithType) GetData() interface{} {
	switch t.TensorType {
	case "float32":
		return t.Tensor.(*Tensor[float32]).GetData()
	case "float64":
		return t.Tensor.(*Tensor[float64]).GetData()
	case "int8":
		return t.Tensor.(*Tensor[int8]).GetData()
	case "int16":
		return t.Tensor.(*Tensor[int16]).GetData()
	case "int32":
		return t.Tensor.(*Tensor[int32]).GetData()
	case "int64":
		return t.Tensor.(*Tensor[int64]).GetData()
	case "uint8":
		return t.Tensor.(*Tensor[uint8]).GetData()
	case "uint16":
		return t.Tensor.(*Tensor[uint16]).GetData()
	case "uint32":
		return t.Tensor.(*Tensor[uint32]).GetData()
	case "uint64":
		return t.Tensor.(*Tensor[uint64]).GetData()
	case "bool":
		return t.Tensor.(*Tensor[bool]).GetData()
	}
	return nil
}

func (t *TensorWithType) Destroy() error {
	switch t.TensorType {
	case "float32":
		return t.Tensor.(*Tensor[float32]).Destroy()
	case "float64":
		return t.Tensor.(*Tensor[float64]).Destroy()
	case "int8":
		return t.Tensor.(*Tensor[int8]).Destroy()
	case "int16":
		return t.Tensor.(*Tensor[int16]).Destroy()
	case "int32":
		return t.Tensor.(*Tensor[int32]).Destroy()
	case "int64":
		return t.Tensor.(*Tensor[int64]).Destroy()
	case "uint8":
		return t.Tensor.(*Tensor[uint8]).Destroy()
	case "uint16":
		return t.Tensor.(*Tensor[uint16]).Destroy()
	case "uint32":
		return t.Tensor.(*Tensor[uint32]).Destroy()
	case "uint64":
		return t.Tensor.(*Tensor[uint64]).Destroy()
	case "bool":
		return t.Tensor.(*Tensor[bool]).Destroy()
	}
	return nil
}

type Tensor[T TensorData] struct {
	// The shape of the tensor
	shape Shape
	// The go slice containing the flattened data that backs the ONNX tensor.
	data []T
	// The underlying ONNX value we use with the C API.
	ortValue *C.OrtValue
}

// Cleans up and frees the memory associated with this tensor.
func (t *Tensor[_]) Destroy() error {
	C.ReleaseOrtValue(t.ortValue)
	t.ortValue = nil
	t.data = nil
	t.shape = nil
	return nil
}

// Returns the slice containing the tensor's underlying data. The contents of
// the slice can be read or written to get or set the tensor's contents.
func (t *Tensor[T]) GetData() []T {
	return t.data
}

// Returns the shape of the tensor. The returned shape is only a copy;
// modifying this does *not* change the shape of the underlying tensor.
// (Modifying the tensor's shape can only be accomplished by Destroying and
// recreating the tensor with the same data.)
func (t *Tensor[_]) GetShape() Shape {
	return t.shape.Clone()
}

// Makes a deep copy of the tensor, including its ONNXRuntime value. The Tensor
// returned by this function must be destroyed when no longer needed.
func (t *Tensor[T]) Clone() (*Tensor[T], error) {
	// TODO: Implement Tensor.Clone()
	return nil, fmt.Errorf("Tensor.Clone is not yet implemented")
}

// Creates a new empty tensor with the given shape. The shape provided to this
// function is copied, and is no longer needed after this function returns.
func NewEmptyTensor[T TensorData](s Shape) (*Tensor[T], error) {
	elementCount := s.FlattenedSize()
	if elementCount == 0 {
		return nil, fmt.Errorf("Got invalid shape containing 0 elements")
	}
	data := make([]T, elementCount)
	return NewTensor(s, data)
}

func NewEmptyTensorWithType(tensorType string, s Shape) (interface{}, error) {
	elementCount := s.FlattenedSize()
	if elementCount == 0 {
		return nil, fmt.Errorf("Got invalid shape containing 0 elements")
	}
	switch tensorType {
	case "float32":
		data := make([]float32, elementCount)
		return NewTensor(s, data)
	case "float64":
		data := make([]float64, elementCount)
		return NewTensor(s, data)
	case "int8":
		data := make([]int8, elementCount)
		return NewTensor(s, data)
	case "int16":
		data := make([]int16, elementCount)
		return NewTensor(s, data)
	case "int32":
		data := make([]int32, elementCount)
		return NewTensor(s, data)
	case "int64":
		data := make([]int64, elementCount)
		return NewTensor(s, data)
	case "uint8":
		data := make([]uint8, elementCount)
		return NewTensor(s, data)
	case "uint16":
		data := make([]uint16, elementCount)
		return NewTensor(s, data)
	case "uint32":
		data := make([]uint32, elementCount)
		return NewTensor(s, data)
	case "uint64":
		data := make([]uint64, elementCount)
		return NewTensor(s, data)
	case "bool":
		data := make([]bool, elementCount)
		return NewTensor(s, data)
	default:
		return nil, fmt.Errorf("Unsupported tensor type: %s", tensorType)
	}
}

// Creates a new tensor backed by an existing data slice. The shape provided to
// this function is copied, and is no longer needed after this function
// returns. If the data slice is longer than s.FlattenedSize(), then only the
// first portion of the data will be used.
func NewTensor[T TensorData](s Shape, data []T) (*Tensor[T], error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}

	elementCount := s.FlattenedSize()
	if elementCount > int64(len(data)) {
		return nil, fmt.Errorf("The tensor's shape (%s) requires %d "+
			"elements, but only %d were provided\n", s, elementCount,
			len(data))
	}
	var ortValue *C.OrtValue
	dataType := GetTensorElementDataType[T]()
	dataSize := unsafe.Sizeof(data[0]) * uintptr(elementCount)

	status := C.CreateOrtTensorWithShape(unsafe.Pointer(&data[0]),
		C.size_t(dataSize), (*C.int64_t)(unsafe.Pointer(&s[0])),
		C.int64_t(len(s)), ortMemoryInfo, dataType, &ortValue)
	if status != nil {
		return nil, fmt.Errorf("ORT API error creating tensor: %s",
			statusToError(status))
	}

	toReturn := Tensor[T]{
		data:     data[0:elementCount],
		shape:    s.Clone(),
		ortValue: ortValue,
	}
	// TODO: Set a finalizer on new Tensors to hopefully prevent careless
	// memory leaks.
	// - Idea: use a "destroyable" interface?
	return &toReturn, nil
}

// A wrapper around the OrtSession C struct. Requires the user to maintain all
// input and output tensors, and to use the same data type for input and output
// tensors.
type Session struct {
	ortSession *C.OrtSession
	// We convert the tensor names to C strings only once, and keep them around
	// here for future calls to Run().
	inputNames  []*C.char
	outputNames []*C.char
	// We only actually keep around the OrtValue pointers from the tensors.
	inputs  []*C.OrtValue
	outputs []*C.OrtValue

	inputCount  C.size_t
	outputCount C.size_t
}

type SessionV2 struct {
	ortSession *C.OrtSession
	// We convert the tensor names to C strings only once, and keep them around
	// here for future calls to Run().
	inputNames  **C.char
	outputNames **C.char

	inputCount  C.int
	outputCount C.int
	// We only actually keep around the OrtValue pointers from the tensors.
	inputs  []*C.OrtValue
	outputs []*C.OrtValue
}

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithONNXData[IT TensorData, OT TensorData](onnxData []byte, inputNames,
	outputNames []string, inputs []*Tensor[IT], outputs []*Tensor[OT]) (*Session, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("No inputs were provided")
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("No outputs were provided")
	}
	if len(inputs) != len(inputNames) {
		return nil, fmt.Errorf("Got %d input tensors, but %d input names",
			len(inputs), len(inputNames))
	}
	if len(outputs) != len(outputNames) {
		return nil, fmt.Errorf("Got %d output tensors, but %d output names",
			len(outputs), len(outputNames))
	}

	var ortSession *C.OrtSession
	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &ortSession)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	// ONNXRuntime copies the file content unless a specific flag is provided
	// when creating the session (and we don't provide it!)

	// Collect the inputs and outputs, along with their names, into a format
	// more convenient for passing to the Run() function in the C API.
	cInputNames := make([]*C.char, len(inputNames))
	cOutputNames := make([]*C.char, len(outputNames))
	for i, v := range inputNames {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range outputNames {
		cOutputNames[i] = C.CString(v)
	}
	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.ortValue
	}
	return &Session{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}

func checkInputsOutputs(inputs []*TensorWithType, outputs []*TensorWithType, inputNames, outputNames []string) error {
	if len(inputs) == 0 {
		return fmt.Errorf("No inputs were provided")
	}
	if len(outputs) == 0 {
		return fmt.Errorf("No outputs were provided")
	}
	if len(inputs) != len(inputNames) {
		return fmt.Errorf("Got %d input tensors, but %d input names", len(inputs), len(inputNames))
	}
	if len(outputs) != len(outputNames) {
		return fmt.Errorf("Got %d output tensors, but %d output names", len(outputs), len(outputNames))
	}

	return nil
}

func convertTensors(tensors []*TensorWithType) []*C.OrtValue {
	ortTensors := make([]*C.OrtValue, len(tensors))
	for i, v := range tensors {
		switch v.TensorType {
		case "float32":
			ortTensors[i] = v.Tensor.(*Tensor[float32]).ortValue
		case "float64":
			ortTensors[i] = v.Tensor.(*Tensor[float64]).ortValue
		case "int8":
			ortTensors[i] = v.Tensor.(*Tensor[int8]).ortValue
		case "int16":
			ortTensors[i] = v.Tensor.(*Tensor[int16]).ortValue
		case "int32":
			ortTensors[i] = v.Tensor.(*Tensor[int32]).ortValue
		case "int64":
			ortTensors[i] = v.Tensor.(*Tensor[int64]).ortValue
		case "uint8":
			ortTensors[i] = v.Tensor.(*Tensor[uint8]).ortValue
		case "uint16":
			ortTensors[i] = v.Tensor.(*Tensor[uint16]).ortValue
		case "uint32":
			ortTensors[i] = v.Tensor.(*Tensor[uint32]).ortValue
		case "uint64":
			ortTensors[i] = v.Tensor.(*Tensor[uint64]).ortValue
		case "bool":
			ortTensors[i] = v.Tensor.(*Tensor[bool]).ortValue
		}
	}
	return ortTensors
}

func convertNames(names []string) []*C.char {
	cNames := make([]*C.char, len(names))
	for i, v := range names {
		cNames[i] = C.CString(v)
	}
	return cNames
}

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionONNXDataWithType(onnxData []byte, inputNames,
	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType) (*Session, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}

	err := checkInputsOutputs(inputs, outputs, inputNames, outputNames)
	if err != nil {
		return nil, err
	}

	var ortSession *C.OrtSession
	status := C.CreateSession(unsafe.Pointer(&(onnxData[0])),
		C.size_t(len(onnxData)), ortEnv, &ortSession)
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
		if deviceType == "coreml" {
			cFunctionName := C.CString("OrtSessionOptionsAppendExecutionProvider_CoreML")
			defer C.free(unsafe.Pointer(cFunctionName))
			appendOptionsProc := C.dlsym(libraryHandle, cFunctionName)
			C.CallAppendOptionsFunction(appendOptionsProc, options)
		} else if deviceType == "cuda" {
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

// Runs the session, updating the contents of the output tensors on success. Caller need to handle release inputs / outputs
func (s *SessionV2) Run(inputs []*TensorWithType, outputs []*TensorWithType) error {

	s.inputs = convertTensors(inputs)
	s.outputs = convertTensors(outputs)

	// fmt.Print("inputCount: ", s.inputCount, "\n")
	// fmt.Print("outputCount: ", s.outputCount, "\n")
	// fmt.Printf("inputNames: %v\n", s.inputNames)
	// fmt.Printf("outputNames: %v\n", s.outputNames)

	status := C.RunOrtSession(s.ortSession, &s.inputs[0], s.inputNames,
		s.inputCount, &s.outputs[0], s.outputNames, s.outputCount)
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}

func (s *SessionV2) RunV2(inputs []*TensorWithType, outputs []*TensorWithType) (err error) {

	s.inputs = convertTensors(inputs)
	// s.outputs = convertTensors(outputs)
	s.outputs = make([]*C.OrtValue, len(outputs))

	// fmt.Print("inputCount: ", s.inputCount, "\n")
	// fmt.Print("outputCount: ", s.outputCount, "\n")
	// fmt.Printf("inputNames: %v\n", s.inputNames)
	// fmt.Printf("outputNames: %v\n", s.outputNames)

	status := C.RunOrtSession(s.ortSession, &s.inputs[0], s.inputNames,
		s.inputCount, &s.outputs[0], s.outputNames, s.outputCount)
	if status != nil {
		return fmt.Errorf("error running network: %w", statusToError(status))
	}

	for i := 0; i < len(outputs); i++ {
		// fmt.Printf("s.outputs[i]: %v\n", s.outputs[i])
		// create new Tensor and assign it to outputs[i].Tensor

		numDims := int(C.GetTensorNumDimensions(s.outputs[i]))
		shape := make([]int64, numDims)
		elementCount := int64(1)
		for j := 0; j < numDims; j++ {
			shape[j] = int64(C.GetTensorDimensions(s.outputs[i], C.size_t(j)))
			elementCount *= shape[j]
		}

		// fmt.Printf("elementCount: %v\n", elementCount)
		// shape := make([]int64, 4)

		// get data type
		dataType := C.GetTensorElementType(s.outputs[i])

		// switch
		// fmt.Printf("dataType: %v\n", dataType)
		switch dataType {
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float32, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float64, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int8, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint8, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int16, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint16, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int32, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint32, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int64, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint64, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]string, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]bool, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float16, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]complex64, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]complex128, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]bfloat16, elementCount), shape)
		default:
			return fmt.Errorf("unsupported data type: %v", dataType)
		}
		// C.ReleaseOrtValue(s.outputs[i])
	}
	if err != nil {
		return fmt.Errorf("error getting tensor: %w", err)
	}
	return nil
}

func GetTensor[T TensorData](value *C.OrtValue, elementCount int64, data []T, shape []int64) (*Tensor[T], error) {
	status := C.GetTensorMutableData(value, unsafe.Pointer(&data[0]))
	if status != nil {
		return nil, fmt.Errorf("error getting data: %w", statusToError(status))
	}
	return &Tensor[T]{
		data:     data[0:elementCount],
		shape:    shape,
		ortValue: value,
	}, nil
}

func (s *SessionV2) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	C.FreeNames(s.inputNames, s.inputCount)
	C.FreeNames(s.outputNames, s.outputCount)
	s.inputCount = 0
	s.outputCount = 0
	s.inputNames = nil
	s.outputNames = nil
	s.inputs = nil
	s.outputs = nil
	return nil
}

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

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithPathWithTypeWithCUDA(path string, inputNames,
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

	cudaDeviceId := C.int(0)
	if len(opts) > 0 {
		// convert opts[0] from string to int
		deviceInt, err := strconv.Atoi(opts[0])
		if err != nil {
			deviceInt = 0
		}
		cudaDeviceId = C.int(deviceInt)
	}

	status := C.CreateSessionPathWithCUDA(modelPath, ortEnv, &ortSession, cudaDeviceId)
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

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithPathWithType(path string, inputNames,
	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType) (*Session, error) {
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
	status := C.CreateSessionPath(modelPath, ortEnv, &ortSession)
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

// The same as NewSession, but takes a slice of bytes containing the .onnx
// network rather than a file path.
func NewSessionWithPath[IT TensorData, OT TensorData](path string, inputNames,
	outputNames []string, inputs []*Tensor[IT], outputs []*Tensor[OT]) (*Session, error) {
	if !IsInitialized() {
		return nil, NotInitializedError
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("No inputs were provided")
	}
	if len(outputs) == 0 {
		return nil, fmt.Errorf("No outputs were provided")
	}
	if len(inputs) != len(inputNames) {
		return nil, fmt.Errorf("Got %d input tensors, but %d input names",
			len(inputs), len(inputNames))
	}
	if len(outputs) != len(outputNames) {
		return nil, fmt.Errorf("Got %d output tensors, but %d output names",
			len(outputs), len(outputNames))
	}

	var ortSession *C.OrtSession
	modelPath := C.CString(path)
	defer C.free(unsafe.Pointer(modelPath))
	status := C.CreateSessionPath(modelPath, ortEnv, &ortSession)
	if status != nil {
		return nil, fmt.Errorf("Error creating session: %w",
			statusToError(status))
	}
	// ONNXRuntime copies the file content unless a specific flag is provided
	// when creating the session (and we don't provide it!)

	// Collect the inputs and outputs, along with their names, into a format
	// more convenient for passing to the Run() function in the C API.
	cInputNames := make([]*C.char, len(inputNames))
	cOutputNames := make([]*C.char, len(outputNames))
	for i, v := range inputNames {
		cInputNames[i] = C.CString(v)
	}
	for i, v := range outputNames {
		cOutputNames[i] = C.CString(v)
	}
	inputOrtTensors := make([]*C.OrtValue, len(inputs))
	outputOrtTensors := make([]*C.OrtValue, len(outputs))
	for i, v := range inputs {
		inputOrtTensors[i] = v.ortValue
	}
	for i, v := range outputs {
		outputOrtTensors[i] = v.ortValue
	}
	return &Session{
		ortSession:  ortSession,
		inputNames:  cInputNames,
		outputNames: cOutputNames,
		inputs:      inputOrtTensors,
		outputs:     outputOrtTensors,
	}, nil
}

// Loads the ONNX network at the given path, and initializes a Session
// instance. If this returns successfully, the caller must call Destroy() on
// the returned session when it is no longer needed. We require the user to
// provide the input and output tensors and names at this point, in order to
// not need to re-allocate them every time Run() is called. The user instead
// can just update or access the input/output tensor data after calling Run().
// The input and output tensors MUST outlive this session, and calling
// session.Destroy() will not destroy the input or output tensors.
func NewSession[IT TensorData, OT TensorData](onnxFilePath string, inputNames,
	outputNames []string, inputs []*Tensor[IT], outputs []*Tensor[OT]) (*Session, error) {

	toReturn, e := NewSessionWithPath(onnxFilePath, inputNames, outputNames, inputs, outputs)
	// fileContent, e := os.ReadFile(onnxFilePath)
	// if e != nil {
	// 	return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
	// }

	// toReturn, e := NewSessionWithONNXData(fileContent, inputNames,
	// 	outputNames, inputs, outputs)
	if e != nil {
		return nil, fmt.Errorf("Error creating session from %s: %w",
			onnxFilePath, e)
	}
	return toReturn, nil
}

func IsCoreMLAvailable() bool {
	status := C.IsCoreMLAvailable()
	if status != nil {
		fmt.Printf("Error checking CoreML availability: %s\n", statusToError(status))
		return false
	}
	return true
}

func IsCUDAAvailable(deviceId int) bool {
	status := C.IsCUDAAvailable(C.int(deviceId))
	if status != nil {
		fmt.Printf("Error checking CUDA availability: %s\n", statusToError(status))
		return false
	}
	return true
}

// func NewSessionWithTypeWithCoreML(onnxFilePath string, inputNames,
// 	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType, opts ...string) (*Session, error) {

// 	toReturn, e := NewSessionWithPathWithTypeWithCoreML(onnxFilePath, inputNames, outputNames, inputs, outputs, opts...)
// 	// fileContent, e := os.ReadFile(onnxFilePath)
// 	// if e != nil {
// 	// 	return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
// 	// }

// 	// toReturn, e := NewSessionWithONNXData(fileContent, inputNames,
// 	// 	outputNames, inputs, outputs)
// 	if e != nil {
// 		return nil, fmt.Errorf("Error creating session from %s: %w",
// 			onnxFilePath, e)
// 	}
// 	return toReturn, nil
// }

// func NewSessionWithTypeWithCUDA(onnxFilePath string, inputNames,
// 	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType, opts ...string) (*Session, error) {

// 	toReturn, e := NewSessionWithPathWithTypeWithCUDA(onnxFilePath, inputNames, outputNames, inputs, outputs, opts...)
// 	// fileContent, e := os.ReadFile(onnxFilePath)
// 	// if e != nil {
// 	// 	return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
// 	// }

// 	// toReturn, e := NewSessionWithONNXData(fileContent, inputNames,
// 	// 	outputNames, inputs, outputs)
// 	if e != nil {
// 		return nil, fmt.Errorf("Error creating session from %s: %w",
// 			onnxFilePath, e)
// 	}
// 	return toReturn, nil
// }

// func NewSessionWithType(onnxFilePath string, inputNames,
// 	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType) (*Session, error) {

// 	toReturn, e := NewSessionWithPathWithType(onnxFilePath, inputNames, outputNames, inputs, outputs)
// 	// fileContent, e := os.ReadFile(onnxFilePath)
// 	// if e != nil {
// 	// 	return nil, fmt.Errorf("Error reading %s: %w", onnxFilePath, e)
// 	// }

// 	// toReturn, e := NewSessionWithONNXData(fileContent, inputNames,
// 	// 	outputNames, inputs, outputs)
// 	if e != nil {
// 		return nil, fmt.Errorf("Error creating session from %s: %w",
// 			onnxFilePath, e)
// 	}
// 	return toReturn, nil
// }

func (s *Session) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	for i := range s.inputNames {
		C.free(unsafe.Pointer(s.inputNames[i]))
	}
	s.inputNames = nil
	for i := range s.outputNames {
		C.free(unsafe.Pointer(s.outputNames[i]))
	}
	s.outputNames = nil
	s.inputs = nil
	s.outputs = nil
	return nil
}

// Runs the session, updating the contents of the output tensors on success.
func (s *Session) Run() error {
	status := C.RunOrtSession(s.ortSession, &s.inputs[0], &s.inputNames[0],
		C.int(len(s.inputs)), &s.outputs[0], &s.outputNames[0],
		C.int(len(s.outputs)))
	if status != nil {
		return fmt.Errorf("Error running network: %w", statusToError(status))
	}
	return nil
}
