//go:build !mac

package onnxruntime_go

func SetupCoreMLExecutionProvider(options *_Ctype_struct_OrtSessionOptions) {

}

func NewSessionWithPathWithTypeWithCoreML(path string, inputNames,
	outputNames []string, inputs []*TensorWithType, outputs []*TensorWithType, opts ...string) (*Session, error) {
	return nil, nil
}
