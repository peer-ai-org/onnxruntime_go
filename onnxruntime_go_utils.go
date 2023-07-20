package onnxruntime_go

import (
	"math"
	"math/rand"
	"sort"
)

type Log struct {
	Value float64
	Index int
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
