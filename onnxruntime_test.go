package onnxruntime_go

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func downloadFile(filepath string, url string) (err error) {

	// Create the file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		// log.Errorf("bad status: %s", resp.Status)
		return
	}

	// Writer the body to file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}

	return nil
}

func Unzip(src, dest string) error {
	r, err := zip.OpenReader(src)
	if err != nil {
		return err
	}
	defer func() {
		if err := r.Close(); err != nil {
			panic(err)
		}
	}()

	err = os.MkdirAll(dest, 0755)
	if err != nil {
		// log.Debugf("error creating directory: %s", err)
	}

	// Closure to address file descriptors issue with all the deferred .Close() methods
	extractAndWriteFile := func(f *zip.File) error {
		rc, err := f.Open()
		if err != nil {
			return err
		}
		defer func() {
			if err := rc.Close(); err != nil {
				panic(err)
			}
		}()

		path := filepath.Join(dest, f.Name)
		// path := f.Name

		// Check for ZipSlip (Directory traversal)
		if !strings.HasPrefix(path, filepath.Clean(dest)+string(os.PathSeparator)) {
			// log.Errorf("illegal file path: %s", path)
			return fmt.Errorf("illegal file path: %s", path)
		}

		if f.FileInfo().IsDir() {
			os.MkdirAll(path, f.Mode())
		} else {
			os.MkdirAll(filepath.Dir(path), f.Mode())
			f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
			if err != nil {
				return err
			}
			defer func() {
				if err := f.Close(); err != nil {
					// log.Error(f)
					return
				}
			}()

			_, err = io.Copy(f, rc)
			if err != nil {
				return err
			}
		}
		return nil
	}

	for _, f := range r.File {
		err := extractAndWriteFile(f)
		if err != nil {
			return err
		}
	}

	return nil
}

func ExtractTarGz(fname string, dest string) error {
	r, err := os.Open(fname)
	if err != nil {
		// log.Errorf(err.Error())
		return err
	}
	defer r.Close()

	uncompressedStream, err := gzip.NewReader(r)
	if err != nil {
		// log.Error("ExtractTarGz: NewReader failed")
		return err
	}

	tarReader := tar.NewReader(uncompressedStream)

	for {
		header, err := tarReader.Next()

		if err == io.EOF {
			break
		}

		if err != nil {
			// log.Errorf("ExtractTarGz: Next() failed: %s", err.Error())
			return err
		}

		path := filepath.Join(dest, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(path, 0755); err != nil {
				// log.Errorf("ExtractTarGz: MkdirAll() failed: %s", err.Error())
				return err
			}
		default:
			outFile, err := os.Create(path)
			if err != nil {
				// log.Errorf("ExtractTarGz: Create() failed: %s", err.Error())
				return err
			}
			if _, err := io.Copy(outFile, tarReader); err != nil {
				// log.Errorf("ExtractTarGz: Copy() failed: %s", err.Error())
				return err
			}
			outFile.Close()
		}

	}
	return nil
}

// This type is read from JSON and used to determine the inputs and expected
// outputs for an ONNX network.
type testInputsInfo struct {
	InputShape            []int64   `json:"input_shape"`
	FlattenedInput        []float32 `json:"flattened_input"`
	OutputShape           []int64   `json:"output_shape"`
	FlattenedOutput       []float32 `json:"flattened_output"`
	CoreMLFlattenedOutput []float32 `json:"coreml_flattened_output"`
}

type testExternalInputsInfo struct {
	InputShape                   []int64   `json:"latent_model_input_shape"`
	FlattenedInput               []float32 `json:"latent_model_input"`
	EncoderHiddenStatesShape     []int64   `json:"encoder_hidden_states_shape"`
	FlattenedEncoderHiddenStates []float32 `json:"encoder_hidden_states"`
	TShape                       []int64   `json:"t_shape"`
	FlattenedT                   []float32 `json:"t"`
	OutputShape                  []int64   `json:"noise_pred_shape"`
	FlattenedOutput              []float32 `json:"noise_pred"`
}

func InitONNXEnv(cuda bool) error {
	// 386, amd64, arm, arm64, ppc64, ppc64le, mips, mipsle, mips64, mips64le, s390x, s390xle
	var name string
	GOOS := runtime.GOOS
	// GOOS = "linux"
	version := "1.14.1"

	homeDir, err := os.UserHomeDir()
	if err != nil {
		// log.Fatal(err)
		return err
	}
	peerAIPath := filepath.Join(homeDir, ".peerai")

	if GOOS == "windows" {
		// check if file exists
		if cuda {
			name = fmt.Sprintf("onnxruntime-win-x64-gpu-%s", version)
		} else {
			name = fmt.Sprintf("onnxruntime-win-x64-%s", version)
		}
		folderPath := filepath.Join(peerAIPath, name)
		filePath := filepath.Join(peerAIPath, fmt.Sprintf("%s.zip", name))
		if _, err := os.Stat(folderPath); err != nil {
			// path/to/whatever not exists
			// download window
			err = downloadFile(filePath, fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/%s.zip", version, name))
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
			// extract
			err = Unzip(filePath, folderPath)
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
		}
		SetSharedLibraryPath(filepath.Join(folderPath, name, "lib", "onnxruntime.dll"))
	} else if GOOS == "darwin" {
		// ex, err := os.Executable()
		// if err != nil {
		// 	panic(err)
		// }
		// exPath := filepath.Dir(ex)
		// // fmt.Println(exPath)
		// runtime := fmt.Sprintf("libonnxruntime.%s.dylib", version)
		// // check if exPath/runtime exists
		// if _, err := os.Stat(filepath.Join(exPath, runtime)); err == nil {
		// 	SetSharedLibraryPath(filepath.Join(exPath, runtime))
		// 	err = ort.InitializeEnvironment()
		// 	if err != nil {
		// 		log.Error(err)
		// 		return err
		// 	}
		// 	return nil
		// }

		name = fmt.Sprintf("onnxruntime-osx-universal2-%s", version)

		// download darwin
		folderPath := filepath.Join(peerAIPath, name)
		filePath := filepath.Join(peerAIPath, fmt.Sprintf("%s.tgz", name))
		if _, err := os.Stat(folderPath); err != nil {
			// path/to/whatever not exists
			// download window
			err = downloadFile(filePath, fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/%s.tgz", version, name))
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
			// extract
			err = ExtractTarGz(filePath, peerAIPath)
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
		}
		SetSharedLibraryPath(filepath.Join(peerAIPath, name, "lib", fmt.Sprintf("libonnxruntime.%s.dylib", version)))
	} else { // assume linux
		// download linux
		// check arch
		if runtime.GOARCH == "amd64" {
			if cuda {
				name = fmt.Sprintf("onnxruntime-linux-x64-gpu-%s", version)
			} else {
				name = fmt.Sprintf("onnxruntime-linux-x64-%s", version)
			}
		} else if runtime.GOARCH == "arm64" {
			name = fmt.Sprintf("onnxruntime-linux-aarch64-%s", version)
		} else {
			return errors.New("unsupported architecture")
		}
		folderPath := filepath.Join(peerAIPath, name)
		filePath := filepath.Join(peerAIPath, fmt.Sprintf("%s.tgz", name))
		if _, err := os.Stat(folderPath); err != nil {
			// path/to/whatever not exists
			// download window
			err = downloadFile(filePath, fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/%s.tgz", version, name))
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
			// extract
			err = ExtractTarGz(filePath, peerAIPath)
			if err != nil {
				// log.Debugf(err.Error())
				return err
			}
		}
		SetSharedLibraryPath(filepath.Join(peerAIPath, name, "lib", fmt.Sprintf("libonnxruntime.so.%s", version)))
	}
	err = InitializeEnvironment()
	if err != nil {
		// log.Error(err)
		return err
	}
	return nil
}

// func TestIsCoreMLAvailable(t *testing.T) {
// 	e := InitONNXEnv(false)
// 	defer func() {
// 		e := DestroyEnvironment()
// 		if e != nil {
// 			t.Logf("Error cleaning up environment: %s\n", e)
// 			t.FailNow()
// 		}
// 	}()
// 	if e != nil {
// 		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
// 		t.FailNow()
// 	}

// 	coreML := IsCoreMLAvailable()
// 	if !coreML {
// 		t.Logf("CoreML should be available, but it isn't")
// 		t.FailNow()
// 	}
// }

// func TestIsCUDAAvailable(t *testing.T) {
// 	e := InitONNXEnv(true)
// 	defer func() {
// 		e := DestroyEnvironment()
// 		if e != nil {
// 			t.Logf("Error cleaning up environment: %s\n", e)
// 			t.FailNow()
// 		}
// 	}()
// 	if e != nil {
// 		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
// 		t.FailNow()
// 	}

// 	cuda := IsCUDAAvailable(0)
// 	if !cuda {
// 		t.Logf("CUDA should be available, but it isn't")
// 		t.FailNow()
// 	}
// }

// This must be called prior to running each test.
func TestInitializeRuntime(t *testing.T) {
	e := InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()
	// if runtime.GOOS == "windows" {
	// 	SetSharedLibraryPath("test_data/onnxruntime.dll")
	// } else if runtime.GOOS == "darwin" {
	// 	SetSharedLibraryPath("test_data/darwin/libonnxruntime.dylib")
	// } else {
	// 	if runtime.GOARCH == "arm64" {
	// 		SetSharedLibraryPath("test_data/onnxruntime_arm64.so")
	// 	} else {
	// 		SetSharedLibraryPath("test_data/onnxruntime.so")
	// 	}
	// }
	// e := InitializeEnvironment()
	if e != nil {
		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
		t.FailNow()
	}
}

func TestInitializeRuntimeCUDA(t *testing.T) {
	e := InitONNXEnv(true)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()
	// if runtime.GOOS == "windows" {
	// 	SetSharedLibraryPath("test_data/onnxruntime.dll")
	// } else if runtime.GOOS == "darwin" {
	// 	SetSharedLibraryPath("test_data/darwin/libonnxruntime.dylib")
	// } else {
	// 	if runtime.GOARCH == "arm64" {
	// 		SetSharedLibraryPath("test_data/onnxruntime_arm64.so")
	// 	} else {
	// 		SetSharedLibraryPath("test_data/onnxruntime.so")
	// 	}
	// }
	// e := InitializeEnvironment()
	if e != nil {
		t.Logf("Failed setting up onnxruntime environment: %s\n", e)
		t.FailNow()
	}
}

// Used to obtain the shape
func parseInputsJSON(path string, t *testing.T) *testInputsInfo {
	toReturn := testInputsInfo{}
	f, e := os.Open(path)
	if e != nil {
		t.Logf("Failed opening %s: %s\n", path, e)
		t.FailNow()
	}
	defer f.Close()
	d := json.NewDecoder(f)
	e = d.Decode(&toReturn)
	if e != nil {
		t.Logf("Failed decoding %s: %s\n", path, e)
		t.FailNow()
	}
	return &toReturn
}

func parseExternalInputsJSON(path string, t *testing.T) *testExternalInputsInfo {
	toReturn := testExternalInputsInfo{}
	f, e := os.Open(path)
	if e != nil {
		t.Logf("Failed opening %s: %s\n", path, e)
		t.FailNow()
	}
	defer f.Close()
	d := json.NewDecoder(f)
	e = d.Decode(&toReturn)
	if e != nil {
		t.Logf("Failed decoding %s: %s\n", path, e)
		t.FailNow()
	}
	return &toReturn
}

// Returns an error if any element between a and b don't match.
func floatsEqual(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("Length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
			// Arbitrarily chosen precision.
			if diff >= 0.00000001 {
				return fmt.Errorf("Data element %d doesn't match: %f vs %v",
					i, a[i], b[i])
			}
		}
	}
	return nil
}

func TestTensorTypes(t *testing.T) {
	// It would be nice to compare this, but doing that would require exposing
	// the underlying C types in Go; the testing package doesn't support cgo.
	type myFloat float64
	dataType := GetTensorElementDataType[myFloat]()
	t.Logf("Got data type for float64-based double: %d\n", dataType)
}

func TestCreateTensor(t *testing.T) {
	InitONNXEnv(false)
	defer DestroyEnvironment()
	s := NewShape(1, 2, 3)
	tensor1, e := NewEmptyTensor[uint8](s)
	if e != nil {
		t.Logf("Failed creating %s uint8 tensor: %s\n", s, e)
		t.FailNow()
	}
	defer tensor1.Destroy()
	if len(tensor1.GetData()) != 6 {
		t.Logf("Incorrect data length for tensor1: %d\n",
			len(tensor1.GetData()))
	}
	// Make sure that the underlying tensor created a copy of the shape we
	// passed to NewEmptyTensor.
	s[1] = 3
	if tensor1.GetShape()[1] == s[1] {
		t.Logf("Modifying the original shape incorrectly changed the " +
			"tensor's shape.\n")
		t.FailNow()
	}

	// Try making a tensor with a different data type.
	s = NewShape(2, 5)
	data := []float32{1.0}
	_, e = NewTensor(s, data)
	if e == nil {
		t.Logf("Didn't get error when creating a tensor with too little " +
			"data.\n")
		t.FailNow()
	}
	t.Logf("Got expected error when creating a tensor without enough data: "+
		"%s\n", e)

	// It shouldn't be an error to create a tensor with too *much* underlying
	// data; we'll just use the first portion of it.
	data = []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
	tensor2, e := NewTensor(s, data)
	if e != nil {
		t.Logf("Error creating tensor with data: %s\n", e)
		t.FailNow()
	}
	defer tensor2.Destroy()
	// Make sure the tensor's internal slice only refers to the part we care
	// about, and not the entire slice.
	if len(tensor2.GetData()) != 10 {
		t.Logf("New tensor data contains %d elements, when it should "+
			"contain 10.\n", len(tensor2.GetData()))
		t.FailNow()
	}
}

func makeRange[T TensorData](min, max int) []T {
	a := make([]T, max-min+1)
	// for i := range a {
	// 	a[i] = T(1.0)
	// }
	return a
}

func TestLoadExternalFormatOnnx(t *testing.T) {
	InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	inputs := parseExternalInputsJSON("test_data/example_network_results_external.json", t)
	inputShape := Shape(inputs.InputShape)
	inputTensor, e := NewTensor(inputShape,
		makeRange[float32](0, int(inputShape.FlattenedSize())))
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()

	inputTShape := Shape(inputs.TShape)
	inputTensorT, e := NewTensor(inputTShape,
		inputs.FlattenedT)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensorT.Destroy()

	inputEncShape := Shape(inputs.EncoderHiddenStatesShape)
	inputTensorEnc, e := NewTensor(inputEncShape,
		makeRange[float32](0, int(inputEncShape.FlattenedSize())))
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensorEnc.Destroy()

	outputTensor, e := NewEmptyTensor[float32](Shape(inputs.OutputShape))
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	if _, err := os.Stat("test_data/unet/model.onnx"); err != nil {
		return
	}

	// Set up and run the session.
	session, e := NewSessionWithPathWithType("test_data/unet/model.onnx",
		[]string{"sample", "timestep", "encoder_hidden_states"}, []string{"out_sample"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "float32",
		}, {
			Tensor:     inputTensorT,
			TensorType: "float32",
		}, {
			Tensor:     inputTensorEnc,
			TensorType: "float32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	fmt.Printf("input1 %+v\n", inputTensor.GetData()[0:10])
	fmt.Printf("output1 %+v\n", outputTensor.GetData()[0:10])
	// e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	// if e != nil {
	// 	t.Logf("The neural network didn't produce the correct result: %s\n", e)
	// 	t.FailNow()
	// }
}

func TestLoadExternalFormatOnnxTextEncoder(t *testing.T) {
	InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	// inputs := parseExternalInputsJSON("test_data/example_network_results_external.json", t)
	inputShape := Shape([]int64{1, 77})
	inputTensor, e := NewTensor(inputShape,
		makeRange[int32](0, int(inputShape.FlattenedSize())))
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()

	outputShape := Shape([]int64{1, 77, 768})
	outputTensor, e := NewEmptyTensor[float32](outputShape)
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	outputShape2 := Shape([]int64{1, 768})
	outputTensor2, e := NewEmptyTensor[float32](outputShape2)
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	if _, err := os.Stat("test_data/text_encoder/model.onnx"); err != nil {
		return
	}
	// Set up and run the session.
	session, e := NewSessionWithPathWithType("test_data/text_encoder/model.onnx",
		[]string{"input_ids"}, []string{"last_hidden_state", "pooler_output"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "int32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}, {
			Tensor:     outputTensor2,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	fmt.Printf("input2 %+v\n", inputTensor.GetData()[0:10])
	fmt.Printf("output2 %+v\n", outputTensor.GetData()[0:10])
	// e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	// if e != nil {
	// 	t.Logf("The neural network didn't produce the correct result: %s\n", e)
	// 	t.FailNow()
	// }
}

func TestLoadExternalFormatOnnxVAEDecoder(t *testing.T) {
	InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	// inputs := parseExternalInputsJSON("test_data/example_network_results_external.json", t)
	inputShape := Shape([]int64{1, 4, 64, 64})
	inputTensor, e := NewTensor(inputShape,
		makeRange[float32](0, int(inputShape.FlattenedSize())))
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()

	outputShape := Shape([]int64{1, 3, 512, 512})
	outputTensor, e := NewEmptyTensor[float32](outputShape)
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	if _, err := os.Stat("test_data/vae_decoder/model.onnx"); err != nil {
		return
	}

	// Set up and run the session.
	session, e := NewSessionWithPathWithType("test_data/vae_decoder/model.onnx",
		[]string{"latent_sample"}, []string{"sample"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "float32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	fmt.Printf("input3 %+v\n", inputTensor.GetData()[0:10])
	fmt.Printf("output3 %+v\n", outputTensor.GetData()[0:10])
	// e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	// if e != nil {
	// 	t.Logf("The neural network didn't produce the correct result: %s\n", e)
	// 	t.FailNow()
	// }
}

func TestExampleNetwork(t *testing.T) {
	InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewEmptyTensor[float32](Shape(inputs.OutputShape))
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	// Set up and run the session.
	session, e := NewSessionWithPathWithType("test_data/example_network.onnx",
		[]string{"1x4 Input Vector"}, []string{"1x2 Output Vector"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "float32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	if e != nil {
		t.Logf("The neural network didn't produce the correct result: %s\n", e)
		t.FailNow()
	}
}

func TestExampleNetworkWithCoreML(t *testing.T) {
	InitONNXEnv(false)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewEmptyTensor[float32](Shape(inputs.OutputShape))
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	// Set up and run the session.
	session, e := NewSessionWithPathWithTypeWithCoreML("test_data/example_network.onnx",
		[]string{"1x4 Input Vector"}, []string{"1x2 Output Vector"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "float32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	// fmt.Printf("input4 %+v\n", inputTensor.GetData()[0:4])
	// fmt.Printf("output4 %+v\n", outputTensor.GetData()[0:2])
	// fmt.Printf("output4 %+v\n", inputs.FlattenedOutput[0:2])
	e = floatsEqual(outputTensor.GetData(), inputs.CoreMLFlattenedOutput)
	if e != nil {
		t.Logf("The neural network didn't produce the correct result: %s\n", e)
		t.FailNow()
	}
}

func TestExampleNetworkWithCUDA(t *testing.T) {
	InitONNXEnv(true)
	defer func() {
		e := DestroyEnvironment()
		if e != nil {
			t.Logf("Error cleaning up environment: %s\n", e)
			t.FailNow()
		}
	}()

	// Create input and output tensors
	inputs := parseInputsJSON("test_data/example_network_results.json", t)
	inputTensor, e := NewTensor(Shape(inputs.InputShape),
		inputs.FlattenedInput)
	if e != nil {
		t.Logf("Failed creating input tensor: %s\n", e)
		t.FailNow()
	}
	defer inputTensor.Destroy()
	outputTensor, e := NewEmptyTensor[float32](Shape(inputs.OutputShape))
	if e != nil {
		t.Logf("Failed creating output tensor: %s\n", e)
		t.FailNow()
	}
	defer outputTensor.Destroy()

	// Set up and run the session.
	session, e := NewSessionWithPathWithTypeWithCUDA("test_data/example_network.onnx",
		[]string{"1x4 Input Vector"}, []string{"1x2 Output Vector"},
		[]*TensorWithType{{
			Tensor:     inputTensor,
			TensorType: "float32",
		}}, []*TensorWithType{{
			Tensor:     outputTensor,
			TensorType: "float32",
		}})
	if e != nil {
		t.Logf("Failed creating session: %s\n", e)
		t.FailNow()
	}
	defer session.Destroy()
	e = session.Run()
	if e != nil {
		t.Logf("Failed to run the session: %s\n", e)
		t.FailNow()
	}
	e = floatsEqual(outputTensor.GetData(), inputs.FlattenedOutput)
	if e != nil {
		t.Logf("The neural network didn't produce the correct result: %s\n", e)
		t.FailNow()
	}
}
