// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	"fmt"
	// "math"
	// "strconv"
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

type SessionV3 struct {
	ortSession *C.OrtSession
	// We convert the tensor names to C strings only once, and keep them around
	// here for future calls to Run().
	inputNames  **C.char
	outputNames **C.char

	inputCount  C.int
	outputCount C.int

	inputTypes           **C.char
	inputShapes          **C.int64_t
	inputSymbolicShapes  ***C.char
	inputShapesCount     *C.int64_t
	outputTypes          **C.char
	outputShapes         **C.int64_t
	outputSymbolicShapes ***C.char
	outputShapesCount    *C.int64_t

	// We only actually keep around the OrtValue pointers from the tensors.
	inputs  []*C.OrtValue
	outputs []*C.OrtValue
}

func Reshape(data []float64, shape []int64) [][]float64 {
	totalSize := int64(1)
	for _, dim := range shape {
		totalSize *= dim
	}

	if totalSize != int64(len(data)) {
		panic("Input data size does not match the specified shape")
	}

	result := make([][]float64, shape[0])
	stride := totalSize / shape[0]
	remainingData := data

	for i := int64(0); i < shape[0]; i++ {
		result[i] = make([]float64, stride)
		copy(result[i], remainingData[:stride])
		remainingData = remainingData[stride:]
	}

	return result
}

func convertCStrings(cStrings **C.char, count int) []string {
	var result = make([]string, count)
	// Calculate the size of a pointer in bytes
	size := unsafe.Sizeof(cStrings)
	for i := 0; i < count; i++ {
		// Calculate the pointer to the i-th C string
		cstr := (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cStrings)) + uintptr(i)*size))
		// Convert the C string to a Go string
		str := C.GoString(*cstr)
		result[i] = str
	}
	return result
}

func convertShape(cShapes **C.int64_t, cShapeCounts *C.int64_t, index int) []int64 {
	cShape := (*[1 << 30]*C.int64_t)(unsafe.Pointer(cShapes))
	cArray := (*[1 << 30]C.int64_t)(unsafe.Pointer(cShape[index]))

	cShapeCount := (*[1 << 30]C.int64_t)(unsafe.Pointer(cShapeCounts))

	dims := int(cShapeCount[index])
	shape := make([]int64, dims)
	for i := 0; i < dims; i++ {
		shape[i] = int64(cArray[i])
	}
	return shape
}

func convertSymbolicShape(cShapes ***C.char, cShapeCounts *C.int64_t, index int) []string {
	cShape := (*[1 << 30]*C.char)(unsafe.Pointer(cShapes))
	cArray := (*[1 << 30]*C.char)(unsafe.Pointer(cShape[index]))

	cShapeCount := (*[1 << 30]C.int64_t)(unsafe.Pointer(cShapeCounts))

	dims := int(cShapeCount[index])
	shape := make([]string, dims)
	for i := 0; i < dims; i++ {
		shape[i] = C.GoString(cArray[i])
	}
	return shape
}

func (s *SessionV3) GetInputTypes() []string {
	return convertCStrings(s.inputTypes, int(s.inputCount))
}

func (s *SessionV3) GetInputNames() []string {
	return convertCStrings(s.inputNames, int(s.inputCount))
}

type ShapeType struct {
	Shape         []int64
	SymbolicShape []string
	Type          string
}

func (s *SessionV3) GetInputShapes() (shapeTypes []ShapeType) {
	shapeTypes = make([]ShapeType, s.inputCount)
	size := unsafe.Sizeof(s.inputTypes)
	for i := 0; i < int(s.inputCount); i++ {

		typeStr := (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(s.inputTypes)) + uintptr(i)*size))

		shapeTypes[i] = ShapeType{
			Shape:         convertShape(s.inputShapes, s.inputShapesCount, i),
			SymbolicShape: convertSymbolicShape(s.inputSymbolicShapes, s.inputShapesCount, i),
			Type:          C.GoString(*typeStr),
		}
	}
	return
}

func (s *SessionV3) GetOutputTypes() []string {
	return convertCStrings(s.outputTypes, int(s.outputCount))
}

func (s *SessionV3) GetOutputNames() []string {
	return convertCStrings(s.outputNames, int(s.outputCount))
}

func (s *SessionV3) GetOutputShapes() (shapeTypes []ShapeType) {
	shapeTypes = make([]ShapeType, s.outputCount)
	size := unsafe.Sizeof(s.outputTypes)

	for i := 0; i < int(s.outputCount); i++ {
		typeStr := (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(s.outputTypes)) + uintptr(i)*size))

		shapeTypes[i] = ShapeType{
			Shape:         convertShape(s.outputShapes, s.outputShapesCount, i),
			SymbolicShape: convertSymbolicShape(s.outputSymbolicShapes, s.outputShapesCount, i),
			Type:          C.GoString(*typeStr),
		}
	}
	return
}

func (s *SessionV3) Run(inputs []*TensorWithType) (outputs []*TensorWithType, err error) {

	s.inputs = ConvertTensors(inputs)
	// s.outputs = ConvertTensors(outputs)
	// convert to int
	outputCount := int(s.outputCount)
	outputs = make([]*TensorWithType, outputCount)

	s.outputs = make([]*C.OrtValue, outputCount)

	// fmt.Print("inputCount: ", s.inputCount, "\n")
	// fmt.Print("outputCount: ", s.outputCount, "\n")
	// fmt.Printf("inputNames: %v\n", s.inputNames)
	// fmt.Printf("outputNames: %v\n", s.outputNames)

	status := C.RunOrtSession(s.ortSession, &s.inputs[0], s.inputNames,
		s.inputCount, &s.outputs[0], s.outputNames, s.outputCount)
	if status != nil {
		return nil, fmt.Errorf("error running network: %w", statusToError(status))
	}

	for i := 0; i < outputCount; i++ {
		outputs[i] = &TensorWithType{
			Tensor: nil,
		}
		// fmt.Printf("s.outputs[i]: %v\n", s.outputs[i])
		// create new Tensor and assign it to outputs[i].Tensor

		numDims := int(C.GetTensorNumDimensions(s.outputs[i]))
		shape := make([]int64, numDims)
		elementCount := int64(1)
		for j := 0; j < numDims; j++ {
			shape[j] = int64(C.GetTensorDimensions(s.outputs[i], C.size_t(j)))
			// fmt.Printf("shape[j]: %v\n", shape[j])
			elementCount *= shape[j]
		}
		// fmt.Printf("i,elementCount,numDims:%v,%v,%v\n", i, elementCount, numDims)
		// shape := make([]int64, 4)

		if elementCount == 0 {
			continue
		}

		// get data type
		dataType := C.GetTensorElementType(s.outputs[i])

		// switch
		// fmt.Printf("dataType: %v\n", dataType)
		switch dataType {
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float32, elementCount), shape)
			outputs[i].TensorType = "float32"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float64, elementCount), shape)
			outputs[i].TensorType = "float64"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int8, elementCount), shape)
			outputs[i].TensorType = "int8"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint8, elementCount), shape)
			outputs[i].TensorType = "uint8"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int16, elementCount), shape)
			outputs[i].TensorType = "int16"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint16, elementCount), shape)
			outputs[i].TensorType = "uint16"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int32, elementCount), shape)
			outputs[i].TensorType = "int32"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint32, elementCount), shape)
			outputs[i].TensorType = "uint32"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]int64, elementCount), shape)
			outputs[i].TensorType = "int64"
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]uint64, elementCount), shape)
			outputs[i].TensorType = "uint64"
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]string, elementCount), shape)
		case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
			outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]bool, elementCount), shape)
			outputs[i].TensorType = "bool"
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]float16, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]complex64, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]complex128, elementCount), shape)
		// case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		// 	outputs[i].Tensor, err = GetTensor(s.outputs[i], elementCount, make([]bfloat16, elementCount), shape)
		default:
			return nil, fmt.Errorf("unsupported data type: %v", dataType)
		}
		// C.ReleaseOrtValue(s.outputs[i])
	}
	if err != nil {
		return nil, fmt.Errorf("error getting tensor: %w", err)
	}
	return
}

func (s *SessionV3) Destroy() error {
	if s.ortSession != nil {
		C.ReleaseOrtSession(s.ortSession)
		s.ortSession = nil
	}
	C.FreeNames(s.inputNames, s.inputCount)
	C.FreeNames(s.outputNames, s.outputCount)
	C.FreeSymbolicShapes(s.inputSymbolicShapes, s.inputShapesCount, s.inputCount)
	C.FreeSymbolicShapes(s.outputSymbolicShapes, s.outputShapesCount, s.outputCount)
	C.FreeTypes(s.inputTypes, s.inputCount)
	C.FreeTypes(s.outputTypes, s.outputCount)
	C.FreeShapes(s.inputShapes, s.inputCount)
	C.FreeShapes(s.outputShapes, s.outputCount)
	C.FreeShapesCount(s.inputShapesCount)
	C.FreeShapesCount(s.outputShapesCount)

	// 	void FreeShapeCounts(int64_t *counts);
	// void FreeTypes(char **types, int count);
	// void FreeShapes(int64_t **shapes, int count);
	// void FreeSymbolicShapes(char ***shapes, int64_t* counts, int count);

	s.inputCount = 0
	s.outputCount = 0
	s.inputNames = nil
	s.outputNames = nil
	s.inputTypes = nil
	s.outputTypes = nil
	s.inputShapes = nil
	s.outputShapes = nil
	s.inputSymbolicShapes = nil
	s.outputSymbolicShapes = nil
	s.inputShapesCount = nil
	s.outputShapesCount = nil
	s.inputs = nil
	s.outputs = nil
	return nil
}

type RunV3GenOptions struct {
	MaxTokens           int
	TopP                float64
	Temperature         float64
	EOSTokenID          int
	ReplacementIndexes  []int
	AttentionMaskIndex  int
	UseCacheBranchIndex int
	MergedDecoder       bool
}

func (s *SessionV3) RunDecoder(inputs []*TensorWithType, opt *RunV3GenOptions) (outTokenIds []int64, err error) {
	maxTokens := opt.MaxTokens
	curTokens := 0
	seqLength := int64(1)
	var outputs []*TensorWithType
	for {
		if curTokens == 0 {
			seqLength = inputs[0].GetShape()[1]
		}
		curTokens += 1
		// fmt.Printf("curTokens: %d\n", curTokens)
		// fmt.Printf("inputs: %v\n", inputs)
		// for i := 0; i < len(inputs); i++ {
		// 	fmt.Printf("inputs[%d]: %v\n", i, inputs[i].GetShape())
		// }
		outputs, err = s.Run(inputs)
		// fmt.Printf("outputs: %v\n", outputs)
		// for i := 0; i < len(outputs); i++ {
		// 	fmt.Printf("outputs[%d]: %v\n", i, outputs[i].GetShape())
		// }
		// fmt.Printf("err: %v\n", err)
		if err != nil {
			return nil, err
		}
		data := inputs[0].GetData().([]int64)
		logits := outputs[0].GetData().([]float32)
		logitsShape := outputs[0].GetShape()
		logs := make([]float64, len(logits))
		for i, v := range logits {
			logs[i] = float64(v) / opt.Temperature
		}
		// first token is always the same
		if curTokens == 1 {
			outTokenIds = append(outTokenIds, data...)
		}
		if seqLength > 1 {
			// find Softmax of logs along dim=1
			// change logs to multi-dim array according to logitsShape
			logsT := Reshape(logs, logitsShape[1:]) // remove first dim
			i := seqLength - 1
			logsT[i] = Softmax(logsT[i])
			topPLogsT, topPLogsTI := TakeTopP(logsT[i], opt.TopP)
			sampledIndex := RandomSelect(topPLogsT)
			tokenId := topPLogsTI[sampledIndex]
			outTokenIds = append(outTokenIds, int64(tokenId))
			if tokenId == opt.EOSTokenID {
				break
			}
			if curTokens >= int(maxTokens-1) {
				break
			}
			seqLength = 1
			// seqLength = 1
			// append data tp outTokenIds

			// replace tokenId to inputs[0]
			newData := []int64{outTokenIds[len(outTokenIds)-1]}
			s := []int64{1, 1}
			tensor, err := NewTensor(s, newData)
			if err != nil {
				return nil, err
			}
			inputs[0].Destroy()
			inputs[0] = &TensorWithType{
				Tensor:     tensor,
				TensorType: "int64",
			}
		} else {
			logs = Softmax(logs)
			topPLogs, topPLogsI := TakeTopP(logs, opt.TopP)
			sampledIndex := RandomSelect(topPLogs)
			tokenId := topPLogsI[sampledIndex]
			outTokenIds = append(outTokenIds, int64(tokenId))
			if tokenId == opt.EOSTokenID {
				break
			}
			if curTokens >= int(maxTokens-1) {
				break
			}
			// replace tokenId to inputs[0]
			data[0] = int64(tokenId)
			s := inputs[0].GetShape()
			// fmt.Printf("s: %v\n", s)
			tensor, err := NewTensor(s, data)
			if err != nil {
				return nil, err
			}
			inputs[0].Destroy()
			inputs[0] = &TensorWithType{
				Tensor:     tensor,
				TensorType: "int64",
			}
		}
		// fmt.Printf("opt.AttentionMaskIndex: %v\n", opt.AttentionMaskIndex)
		if opt.AttentionMaskIndex > -1 {
			k := int(opt.AttentionMaskIndex)
			attData := inputs[k].GetData().([]int64)
			attShape := inputs[k].GetShape()
			newAttData := append(attData, int64(1))
			newAttShape := []int64{attShape[0], attShape[1] + 1}
			attTensor, err := NewTensor(newAttShape, newAttData)
			if err != nil {
				return nil, err
			}
			inputs[k].Destroy()
			inputs[k] = &TensorWithType{
				Tensor:     attTensor,
				TensorType: "int64",
			}
		}
		j := 1
		for _, i := range opt.ReplacementIndexes {
			if outputs[j].Tensor != nil {
				inputs[i].Destroy()
				inputs[i] = &TensorWithType{
					Tensor:     outputs[j].Tensor,
					TensorType: outputs[j].TensorType,
				}
			}
			j += 1
		}
	}
	// outputs[0].Destroy()
	// outputs[0] = &TensorWithType{
	// 	Tensor:     outT,
	// 	TensorType: "int64",
	// }
	for j := 0; j < len(outputs); j++ {
		outputs[j].Destroy()
	}
	return
}

func (s *SessionV3) RunMergedDecoder(inputs []*TensorWithType, opt *RunV3GenOptions) (outTokenIds []int64, err error) {
	maxTokens := opt.MaxTokens
	curTokens := 0
	seqLength := int64(1)
	var outputs []*TensorWithType
	for {
		if curTokens == 0 {
			// s := inputs[0].GetShape()
			// fmt.Printf("s: %v\n", s)
			seqLength = inputs[0].GetShape()[1]
			// fmt.Printf("seqLength: %v\n", seqLength)
		}
		if curTokens == 1 {
			var lastIdx int
			if opt.UseCacheBranchIndex > -1 {
				lastIdx = opt.UseCacheBranchIndex
			} else {
				lastIdx = len(inputs) - 1
			}

			// log
			// fmt.Println("lastIdx", lastIdx, "curTokens", curTokens)

			// replace the last item of inputs with bool false
			s := inputs[lastIdx].GetShape()
			tensor, err := NewTensor(s, []bool{true})
			if err != nil {
				return nil, err
			}
			inputs[lastIdx].Destroy()
			inputs[lastIdx] = &TensorWithType{
				Tensor:     tensor,
				TensorType: "bool",
			}

		}
		curTokens += 1
		// fmt.Printf("curTokens: %d\n", curTokens)
		// fmt.Printf("inputs: %v\n", inputs)
		// for i := 0; i < len(inputs); i++ {
		// 	fmt.Printf("inputs[%d]: %v\n", i, inputs[i].GetShape())
		// }
		outputs, err = s.Run(inputs)
		if err != nil {
			return nil, err
		}
		data := inputs[0].GetData().([]int64)
		logits := outputs[0].GetData().([]float32)
		logitsShape := outputs[0].GetShape()
		// fmt.Printf("logitsShape: %v\n", logitsShape)
		logs := make([]float64, len(logits))
		for i, v := range logits {
			logs[i] = float64(v) / opt.Temperature
		}
		// first token is always the same
		if curTokens == 1 {
			outTokenIds = append(outTokenIds, data...)
		}

		if seqLength > 1 {
			// find Softmax of logs along dim=1
			// change logs to multi-dim array according to logitsShape
			logsT := Reshape(logs, logitsShape[1:]) // remove first dim
			i := seqLength - 1
			logsT[i] = Softmax(logsT[i])
			topPLogsT, topPLogsTI := TakeTopP(logsT[i], opt.TopP)
			sampledIndex := RandomSelect(topPLogsT)
			tokenId := topPLogsTI[sampledIndex]
			outTokenIds = append(outTokenIds, int64(tokenId))
			if tokenId == opt.EOSTokenID {
				break
			}
			if curTokens >= int(maxTokens-1) {
				break
			}
			seqLength = 1
			// seqLength = 1
			// append data tp outTokenIds

			// replace tokenId to inputs[0]
			newData := []int64{outTokenIds[len(outTokenIds)-1]}
			s := []int64{1, 1}
			tensor, err := NewTensor(s, newData)
			if err != nil {
				return nil, err
			}
			inputs[0].Destroy()
			inputs[0] = &TensorWithType{
				Tensor:     tensor,
				TensorType: "int64",
			}

			// fmt.Printf("NEW inputs[0]: %v\n", inputs[0].GetShape())
			// fmt.Printf("NEW inputs[1]: %v\n", inputs[1].GetShape())

		} else {
			logs = Softmax(logs)
			// greedily calculate argmax(outputs[0],dim=2)
			// outputs[0].Tensor
			// replace inputs[0] with outputs[0] argmax result
			// calculate argmax(logits) using temperature
			// sort logs by value max to min
			topPLogs, topPLogsI := TakeTopP(logs, opt.TopP)
			// fmt.Printf("topPLogs: %v %v\n", topPLogs[0], topPLogs[1])
			// fmt.Printf("len(topPLogs): %v\n", len(topPLogs))
			// fmt.Printf("topPLogsI: %v\n", topPLogsI[0])

			sampledIndex := RandomSelect(topPLogs)
			// fmt.Printf("sampledIndex: %d\n", sampledIndex)
			tokenId := topPLogsI[sampledIndex]

			// max := logits[0]
			// maxIndex := 0
			// for i, v := range logits {
			// 	if v > max {
			// 		max = v
			// 		maxIndex = i
			// 	}
			// }
			outTokenIds = append(outTokenIds, int64(tokenId))
			// fmt.Printf("tokenId: %d\n", tokenId)
			if tokenId == opt.EOSTokenID {
				break
			}
			if curTokens >= int(maxTokens-1) {
				break
			}
			// replace tokenId to inputs[0]
			data[0] = int64(tokenId)
			s := inputs[0].GetShape()
			tensor, err := NewTensor(s, data)
			if err != nil {
				return nil, err
			}
			inputs[0].Destroy()
			inputs[0] = &TensorWithType{
				Tensor:     tensor,
				TensorType: "int64",
			}
		}

		if opt.AttentionMaskIndex > -1 {
			k := int(opt.AttentionMaskIndex)
			attData := inputs[k].GetData().([]int64)
			attShape := inputs[k].GetShape()
			newAttData := append(attData, int64(1))
			newAttShape := []int64{attShape[0], attShape[1] + 1}
			attTensor, err := NewTensor(newAttShape, newAttData)
			if err != nil {
				return nil, err
			}
			inputs[k].Destroy()
			inputs[k] = &TensorWithType{
				Tensor:     attTensor,
				TensorType: "int64",
			}
		}
		// release and replace inputs[opt.ReplacementIndexes] with outputs[1:end]
		// ozs := outputs[0].GetShape()
		// fmt.Printf("outputs[%d].GetShape(): %v\n", 0, ozs)

		j := 1
		for _, i := range opt.ReplacementIndexes {
			// os := outputs[j].GetShape()
			// fmt.Printf("outputs[%d].GetShape(): %v\n", j, os)
			if outputs[j].Tensor != nil {
				inputs[i].Destroy()
				inputs[i] = &TensorWithType{
					Tensor:     outputs[j].Tensor,
					TensorType: outputs[j].TensorType,
				}
			}
			j += 1
		}
	}
	// outputs[0].Destroy()
	for j := 0; j < len(outputs); j++ {
		outputs[j].Destroy()
	}
	return
}

func (s *SessionV3) RunGen(inputs []*TensorWithType, opt *RunV3GenOptions) (outputs []*TensorWithType, err error) {
	if opt == nil {
		// default is to fill with array from 2 to len(inputs)
		opt = &RunV3GenOptions{}
	}
	// opt.MaxTokens = 128
	// opt.TopP = 0.9
	// opt.Temperature = 1.0
	// opt.EOSTokenID = 50256
	// indexes := make([]int, len(inputs)-2)
	// for i := 0; i < len(inputs)-2; i++ {
	// 	indexes[i] = i + 2
	// }
	// opt.ReplacementIndexes = indexes
	if opt.MaxTokens == 0 {
		opt.MaxTokens = 128
	}
	if opt.TopP == 0 {
		opt.TopP = 0.1
	}
	if opt.Temperature == 0 {
		opt.Temperature = 1.0
	}
	if opt.EOSTokenID == 0 {
		opt.EOSTokenID = 50256
	}
	if opt.ReplacementIndexes == nil {
		indexes := make([]int, len(inputs)-2)
		for i := 0; i < len(inputs)-2; i++ {
			indexes[i] = i + 2
		}
		opt.ReplacementIndexes = indexes
	}

	// run the use_cache_branch == false first
	// then run the use_cache_branch == true
	var outTokenIds []int64
	if opt.MergedDecoder {
		outTokenIds, err = s.RunMergedDecoder(inputs, opt)
	} else {
		outTokenIds, err = s.RunDecoder(inputs, opt)
	}

	if err != nil {
		return nil, err
	}

	outT, err := NewTensor(NewShape(1, int64(len(outTokenIds))), outTokenIds)
	if err != nil {
		return nil, err
	}
	outputs = append(outputs, &TensorWithType{
		Tensor:     outT,
		TensorType: "int64",
	})

	return
}
