// This library wraps the C "onnxruntime" library maintained at
// https://github.com/microsoft/onnxruntime.  It seeks to provide as simple an
// interface as possible to load and run ONNX-format neural networks from
// Go code.
package onnxruntime_go

import (
	// "fmt"
	"math"
	"sort"

	// "os"
	// "github.com/ebitengine/purego"
	"math/rand"
)

type RunV2GenOptions struct {
	MaxTokens          int
	TopP               float64
	Temperature        float64
	EOSTokenID         int
	ReplacementIndexes []int
}

func RandomSelect(probabilities []float64) int {
	sumProbabilities := 0.0
	for _, p := range probabilities {
		sumProbabilities += p
	}

	r := rand.Float64() * sumProbabilities
	// fmt.Printf("probabilities: %v\n", probabilities)
	// fmt.Printf("r: %v\n", r)
	for i, p := range probabilities {
		r -= p
		if r <= 0 {
			return i
		}
	}

	return 0 // return first (most probable) as a fallback
}

// function to calculate softmax
func Softmax(x []float64) []float64 {
	var sum float64
	for i := range x {
		sum += math.Exp(x[i])
	}
	for i := range x {
		x[i] = math.Exp(x[i]) / sum
	}
	return x
}

type Log struct {
	Value float64
	Index int
}

func TakeTopP(logs []float64, topP float64) ([]float64, []int) {
	logsWithIndices := make([]Log, len(logs))
	for i, log := range logs {
		logsWithIndices[i] = Log{log, i}
	}

	sort.Slice(logsWithIndices, func(i, j int) bool {
		return logsWithIndices[i].Value > logsWithIndices[j].Value
	})

	selectedLogs := make([]float64, 0)
	selectedIndices := make([]int, 0)

	sum := 0.0
	for _, log := range logsWithIndices {
		selectedLogs = append(selectedLogs, log.Value)
		selectedIndices = append(selectedIndices, log.Index)
		sum += log.Value

		if sum >= topP {
			break
		}
	}

	return selectedLogs, selectedIndices
}

func (s *SessionV2) RunV2Gen(inputs []*TensorWithType, outputs []*TensorWithType, opt *RunV2GenOptions) (err error) {
	if opt == nil {
		// default is to fill with array from 2 to len(inputs)
		opt = &RunV2GenOptions{}
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

	maxTokens := opt.MaxTokens
	curTokens := 0
	outTokenIds := []int64{}
	for {
		curTokens += 1
		// fmt.Printf("curTokens: %d\n", curTokens)
		err = s.RunV2(inputs, outputs)
		if err != nil {
			return err
		}
		// greedily calculate argmax(outputs[0],dim=2)
		// outputs[0].Tensor
		// replace inputs[0] with outputs[0] argmax result
		data := inputs[0].GetData().([]int64)
		logits := outputs[0].GetData().([]float32)
		// calculate argmax(logits) using temperature

		logs := make([]float64, len(logits))
		for i, v := range logits {
			logs[i] = float64(v) / opt.Temperature
		}

		logs = Softmax(logs)
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
		data[0] = int64(tokenId)
		s := inputs[0].GetShape()
		tensor, err := NewTensor(s, data)
		if err != nil {
			return err
		}
		defer inputs[0].Destroy()
		inputs[0] = &TensorWithType{
			Tensor:     tensor,
			TensorType: "int64",
		}
		// release and replace inputs[opt.ReplacementIndexes] with outputs[1:end]
		j := 1
		for _, i := range opt.ReplacementIndexes {
			defer inputs[i].Destroy()
			inputs[i] = &TensorWithType{
				Tensor:     outputs[j].Tensor,
				TensorType: outputs[j].TensorType,
			}
			j += 1
		}
	}
	defer outputs[0].Destroy()
	outT, err := NewTensor(NewShape(1, int64(len(outTokenIds))), outTokenIds)
	if err != nil {
		return err
	}
	outputs[0] = &TensorWithType{
		Tensor:     outT,
		TensorType: "int64",
	}
	for j := 1; j < len(outputs); j++ {
		defer outputs[j].Destroy()
	}
	return nil
}
