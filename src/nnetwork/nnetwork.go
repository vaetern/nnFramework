package nnetwork

import (
	"encoding/json"
	"errors"
	"fmt"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"time"
)

// NeuralNet contains all of the information
// that defines a trained neural network.
type NeuralNet struct {
	Config  NeuralNetConfig
	WHidden *mat.Dense `json:"WHidden"`
	BHidden *mat.Dense `json:"BHidden"`
	WOut    *mat.Dense `json:"WOut"`
	BOut    *mat.Dense `json:"BOut"`
}

type portableNeuralNet struct{
	nn NeuralNet
	Config  NeuralNetConfig
	WHidden []byte `json:"WHidden"`
	BHidden []byte `json:"BHidden"`
	WOut    []byte `json:"WOut"`
	BOut    []byte `json:"BOut"`
}

// NeuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
	LearningRate  float64
}

func NewNetwork(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{Config: config}
}

// train trains a neural network using backpropagation.
func (nn *NeuralNet) Train(x, y *mat.Dense) error {

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, nil)
	wOut := mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, nil)
	bOut := mat.NewDense(1, nn.Config.OutputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut

	fmt.Println(nn)

	err := storeNetworkToFile(*nn)
	if err != nil{
		log.Println("failed to save trained network to file", err)
	}

	return nil
}


func storeNetworkToFile(net NeuralNet) (err error){
	filename := time.Now().Format("2006-01-1 03-04-05") + " output.json"

	//translate nn to portable version
	bhid,_ := net.BHidden.MarshalBinary()
	bout,_ := net.BOut.MarshalBinary()
	whid,_ := net.WHidden.MarshalBinary()
	wout,_ := net.WOut.MarshalBinary()

	portableNeuralNet := portableNeuralNet{
		nn:net,
		Config:net.Config,
		BHidden:bhid,
		BOut:bout,
		WHidden:whid,
		WOut:wout,
		}

	net.BHidden.MarshalBinary()

	nnJson, _ := json.Marshal(portableNeuralNet)
	err = ioutil.WriteFile("../results/"+filename, nnJson, 0644)
	return err
}

// backpropagate completes the backpropagation method.
func (nn *NeuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	prevErrorValue := .0
	whenToShowOffset := math.Floor(float64(nn.Config.NumEpochs/20))
	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < nn.Config.NumEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.Config.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.Config.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.Config.LearningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.Config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)

		//error calculate
		if math.Mod(float64(i), whenToShowOffset) == 0 || i == (nn.Config.NumEpochs-1){
			var errorValue float64
			errorValue = .0
			for _, e := range errorAtHiddenLayer.RawMatrix().Data {
				errorValue = + e
			}
			dimX, dimY := errorAtHiddenLayer.Dims()
			errorValue = math.Abs(errorValue / float64(dimX*dimY))
			fmt.Printf("epoch:|%6d| error:|%6.25f|%3.2f|\n", i, errorValue, 100-(errorValue/(prevErrorValue/100)))
			prevErrorValue = errorValue
			//xxx
		}
	}

	return nil
}

// predict makes a prediction based on a trained
// neural network.
func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our NeuralNet value
	// represents a trained model.
	if nn.WHidden == nil || nn.WOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.BHidden == nil || nn.BOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.WOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

// sumAlongAxis sums a matrix along a
// particular dimension, preserving the
// other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}
