// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
	// "strconv"

	// "os"
	// "github.com/ebitengine/purego"
	"unsafe"
)

/*
#cgo CFLAGS: -O2 -g
#include "onnxruntime_wrapper.h"
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

func CheckInputsOutputs(inputs []*TensorWithType, outputs []*TensorWithType, inputNames, outputNames []string) error {
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

func ConvertTensors(tensors []*TensorWithType) []*C.OrtValue {
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

func ConvertNames(names []string) []*C.char {
	cNames := make([]*C.char, len(names))
	for i, v := range names {
		cNames[i] = C.CString(v)
	}
	return cNames
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
