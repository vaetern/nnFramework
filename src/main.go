package main

import (
	"encoding/csv"
	"fmt"
	"log"

	"os"
	"strconv"

	"flag"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	nn "nnetwork"
)

func main() {

	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("../data/train.csv")

	var argNumNeurons, argNumEpochs int
	var argRate float64
	var argMode, argPortableFile string
	flag.IntVar(&argNumNeurons, "n", 1, "number of hidden neurons")
	flag.Float64Var(&argRate, "rate", .01, "learning rate")
	flag.IntVar(&argNumEpochs, "e", 1, "number of training epochs")
	flag.StringVar(&argMode, "MODE", "train", "operationg mode")
	flag.StringVar(&argPortableFile, "network-file", "", "source file with trained network params")
	flag.Parse()

	if argMode != "train" && argMode != "predict-from" {
		log.Fatal("-MODE can be only: ", "'train', 'predict-from' <", argMode, ">")
	}

	if argMode == "predict-from"{
		if argPortableFile == ""{
			log.Fatal("-source-file must be specified")
		}
		network := nn.LoadNetworkFromFile(argPortableFile)
		printConfig(network.Config)
		printAccuracy(network)
	}

	if argMode == "train" {
		// Define our network architecture and learning parameters.
		config := nn.NeuralNetConfig{
			InputNeurons:  4,
			OutputNeurons: 3,
			HiddenNeurons: argNumNeurons,
			NumEpochs:     argNumEpochs,
			LearningRate:  argRate,
		}

		printConfig(config)

		// Train the neural network.
		network := nn.NewNetwork(config)
		if err := network.Train(inputs, labels); err != nil {
			log.Fatal(err)
		}
		printAccuracy(network)
	}



}

func printConfig(config nn.NeuralNetConfig){
	fmt.Println("Input  neurons:", config.InputNeurons)
	fmt.Println("Hidden neurons:", config.HiddenNeurons)
	fmt.Println("Output neurons:", config.OutputNeurons)
	fmt.Println("Learning rate:", config.LearningRate)
	fmt.Println("Epochs", config.NumEpochs)
	fmt.Println("")
}

func printAccuracy(network *nn.NeuralNet){
	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("../data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy: %0.2f\n\n", accuracy)
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}
