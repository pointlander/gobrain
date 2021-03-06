// Package gobrain provides basic neural networks algorithms.
package gobrain

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

// FeedForwad struct is used to represent a simple neural network
type FeedForward struct {
	// Number of input, hidden and output nodes
	NInputs, NHiddens, NOutputs int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	InputActivations, HiddenActivations, OutputActivations []float64
	// ElmanRNN contexts
	Contexts [][]float64
	// Weights
	InputWeights, OutputWeights [][]float64
	// Last change in weights for momentum
	InputChanges, OutputChanges [][]float64
	// Set for dropout
	Dropout float64
}

/*
Initialize the neural network;

the 'inputs' value is the number of inputs the network will have,
the 'hiddens' value is the number of hidden nodes and
the 'outputs' value is the number of the outputs of the network.
*/
func (nn *FeedForward) Init(inputs, hiddens, outputs int) {
	nn.NInputs = inputs + 1   // +1 for bias
	nn.NHiddens = hiddens + 1 // +1 for bias
	nn.NOutputs = outputs

	nn.InputActivations = vector(nn.NInputs, 1.0)
	nn.HiddenActivations = vector(nn.NHiddens, 1.0)
	nn.OutputActivations = vector(nn.NOutputs, 1.0)

	nn.InputWeights = matrix(nn.NHiddens, nn.NInputs)
	nn.OutputWeights = matrix(nn.NOutputs, nn.NHiddens)

	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[j][i] = random(-1, 1)
		}
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[j][i] = random(-1, 1)
		}
	}

	nn.InputChanges = matrix(nn.NInputs, nn.NHiddens)
	nn.OutputChanges = matrix(nn.NHiddens, nn.NOutputs)
}

/*
Set the number of contexts to add to the network.

By default the network do not have any context so it is a simple Feed Forward network,
when contexts are added the network behaves like an Elman's SRN (Simple Recurrent Network).

The first parameter (nContexts) is used to indicate the number of contexts to be used,
the second parameter (initValues) can be used to create custom initialized contexts.

If 'initValues' is set, the first parameter 'nContexts' is ignored and
the contexts provided in 'initValues' are used.

When using 'initValues' note that contexts must have the same size of hidden nodes + 1 (bias node).
*/
func (nn *FeedForward) SetContexts(nContexts int, initValues [][]float64) {
	if initValues == nil {
		initValues = make([][]float64, nContexts)

		for i := 0; i < nContexts; i++ {
			initValues[i] = vector(nn.NHiddens, 0.5)
		}
	}

	nn.Contexts = initValues
}

/*
The Update method is used to activate the Neural Network.

Given an array of inputs, it returns an array, of length equivalent of number of outputs, with values ranging from 0 to 1.
*/
func (nn *FeedForward) Update(inputs []float64) []float64 {
	return nn.update(inputs, false)
}

func (nn *FeedForward) update(inputs []float64, train bool) []float64 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NInputs-1; i++ {
		nn.InputActivations[i] = inputs[i]
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		sum := dot64(nn.InputActivations, nn.InputWeights[i])

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		nn.HiddenActivations[i] = sigmoid(sum)

		//http://iamtrask.github.io/2015/07/28/dropout/
		if train && nn.Dropout != 0 {
			if rand.Float64() > 1-nn.Dropout {
				nn.HiddenActivations[i] = 0
			} else {
				nn.HiddenActivations[i] *= 1 / (1 - nn.Dropout)
			}
		}
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			nn.OutputActivations[i] = dot64(nn.HiddenActivations, nn.OutputWeights[i])
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot64(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = sigmoid(sum)
		}
	}

	return nn.OutputActivations
}

func (nn *FeedForward) UpdateWithNoise(inputs []float64, noise [][]float64) []float64 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	if nn.Regression {
		for i := 0; i < nn.NInputs-1; i++ {
			nn.InputActivations[i] = inputs[i] + noise[0][i]
		}
	} else {
		for i := 0; i < nn.NInputs-1; i++ {
			nn.InputActivations[i] = normalize(inputs[i] + noise[0][i])
		}
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		sum := dot64(nn.InputActivations, nn.InputWeights[i])

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		nn.HiddenActivations[i] = normalize(sigmoid(sum) + noise[1][i])
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot64(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = sum + noise[2][i]
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot64(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = normalize(sigmoid(sum) + noise[2][i])
		}
	}

	return nn.OutputActivations
}

/*
The BackPropagate method is used, when training the Neural Network,
to back propagate the errors from network activation.
*/
func (nn *FeedForward) BackPropagate(targets []float64, lRate, mFactor float64) float64 {
	if len(targets) != nn.NOutputs {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := vector(nn.NOutputs, 0.0)
	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			outputDeltas[i] = (targets[i] - nn.OutputActivations[i])
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			outputDeltas[i] = dsigmoid(nn.OutputActivations[i]) * (targets[i] - nn.OutputActivations[i])
		}
	}

	hiddenDeltas := vector(nn.NHiddens, 0.0)
	for i := 0; i < nn.NHiddens; i++ {
		var e float64

		for j := 0; j < nn.NOutputs; j++ {
			e += outputDeltas[j] * nn.OutputWeights[j][i]
		}

		hiddenDeltas[i] = dsigmoid(nn.HiddenActivations[i]) * e
	}

	change := make([]float64, nn.NOutputs)
	for i := 0; i < nn.NHiddens; i++ {
		copy(change, outputDeltas)
		scal64(nn.HiddenActivations[i], change)
		scal64(mFactor, nn.OutputChanges[i])
		axpy64(lRate, change, nn.OutputChanges[i])
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[j][i] = nn.OutputWeights[j][i] + nn.OutputChanges[i][j]
		}
		copy(nn.OutputChanges[i], change)
	}

	change = make([]float64, nn.NHiddens)
	for i := 0; i < nn.NInputs; i++ {
		copy(change, hiddenDeltas)
		scal64(nn.InputActivations[i], change)
		scal64(mFactor, nn.InputChanges[i])
		axpy64(lRate, change, nn.InputChanges[i])
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[j][i] = nn.InputWeights[j][i] + nn.InputChanges[i][j]
		}
		copy(nn.InputChanges[i], change)
	}

	var e float64

	for i := 0; i < len(targets); i++ {
		e += math.Pow(targets[i]-nn.OutputActivations[i], 2)
	}

	return e
}

/*
This method is used to train the Network, it will run the training operation for 'iterations' times
and return the computed errors when training.
*/
func (nn *FeedForward) Train(patterns [][][]float64, iterations int, lRate, mFactor float64, debug bool) []float64 {
	errors := make([]float64, iterations)

	for i := 0; i < iterations; i++ {
		var e float64
		var n int
		for _, p := range patterns {
			nn.update(p[0], true)

			tmp := nn.BackPropagate(p[1], lRate, mFactor)
			e += tmp
			n += len(p[1])
		}

		errors[i] = e / float64(n)

		if debug && i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *FeedForward) Test(patterns [][][]float64) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}

// FeedForwad struct is used to represent a simple neural network
type FeedForward32 struct {
	// Number of input, hidden and output nodes
	NInputs, NHiddens, NOutputs int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	InputActivations, HiddenActivations, OutputActivations []float32
	// ElmanRNN contexts
	Contexts [][]float32
	// Weights
	InputWeights, OutputWeights [][]float32
	// Last change in weights for momentum
	InputChanges, OutputChanges [][]float32
	// Set for dropout
	Dropout float32
	// Activation function
	Activation  func(x float32) float32
	DActivation func(y float32) float32
}

type Activation32 func(x float32) float32

type Context32 struct {
	Iterations     int
	LRate, MFactor float32
	Debug          bool

	Activations []Activation32
}

type Config32 func(context *Context32) *Context32

/*
Initialize the neural network;

the 'inputs' value is the number of inputs the network will have,
the 'hiddens' value is the number of hidden nodes and
the 'outputs' value is the number of the outputs of the network.
*/
func (nn *FeedForward32) Init(inputs, hiddens, outputs int) {
	nn.NInputs = inputs + 1   // +1 for bias
	nn.NHiddens = hiddens + 1 // +1 for bias
	nn.NOutputs = outputs

	nn.InputActivations = vector32(nn.NInputs, 1.0)
	nn.HiddenActivations = vector32(nn.NHiddens, 1.0)
	nn.OutputActivations = vector32(nn.NOutputs, 1.0)

	nn.InputWeights = matrix32(nn.NHiddens, nn.NInputs)
	nn.OutputWeights = matrix32(nn.NOutputs, nn.NHiddens)

	// http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
	scale := float32(math.Sqrt(float64(nn.NInputs)))
	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[j][i] = random32(-1, 1) / scale
		}
	}

	scale = float32(math.Sqrt(float64(nn.NHiddens)))
	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[j][i] = random32(-1, 1) / scale
		}
	}

	nn.InputChanges = matrix32(nn.NInputs, nn.NHiddens)
	nn.OutputChanges = matrix32(nn.NHiddens, nn.NOutputs)

	nn.Activation = sigmoid32
	nn.DActivation = dsigmoid32
}

func (nn *FeedForward32) SetWeights(weights []float32) {
	w := 0
	for i := 0; i < nn.NInputs; i++ {
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[j][i] = weights[w]
			w++
		}
	}

	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[j][i] = weights[w]
			w++
		}
	}
}

func (nn *FeedForward32) SetTanhActivation() {
	nn.Activation = tanh32
	nn.DActivation = dtanh32
}

/*
Set the number of contexts to add to the network.

By default the network do not have any context so it is a simple Feed Forward network,
when contexts are added the network behaves like an Elman's SRN (Simple Recurrent Network).

The first parameter (nContexts) is used to indicate the number of contexts to be used,
the second parameter (initValues) can be used to create custom initialized contexts.

If 'initValues' is set, the first parameter 'nContexts' is ignored and
the contexts provided in 'initValues' are used.

When using 'initValues' note that contexts must have the same size of hidden nodes + 1 (bias node).
*/
func (nn *FeedForward32) SetContexts(nContexts int, initValues [][]float32) {
	if initValues == nil {
		initValues = make([][]float32, nContexts)

		for i := 0; i < nContexts; i++ {
			initValues[i] = vector32(nn.NHiddens, 0.5)
		}
	}

	nn.Contexts = initValues
}

/*
The Update method is used to activate the Neural Network.

Given an array of inputs, it returns an array, of length equivalent of number of outputs, with values ranging from 0 to 1.
*/
func (nn *FeedForward32) Update(inputs []float32) []float32 {
	context := Context32{
		Activations: []Activation32{nn.Activation, nn.Activation},
	}
	if nn.Regression {
		context.Activations[1] = func(x float32) float32 {
			return x
		}
	}
	return nn.update(inputs, &context)
}

func (nn *FeedForward32) update(inputs []float32, context *Context32) []float32 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NInputs-1; i++ {
		nn.InputActivations[i] = inputs[i]
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		sum := dot32(nn.InputActivations, nn.InputWeights[i])

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		nn.HiddenActivations[i] = context.Activations[0](sum)
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	for i := 0; i < nn.NOutputs; i++ {
		sum := dot32(nn.HiddenActivations, nn.OutputWeights[i])

		nn.OutputActivations[i] = context.Activations[1](sum)
	}

	return nn.OutputActivations
}

func (nn *FeedForward32) HalfUpdate(inputs []float32) []float32 {
	if len(inputs) != nn.NHiddens-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		nn.HiddenActivations[i] = inputs[i]
	}

	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			nn.OutputActivations[i] = dot32(nn.HiddenActivations, nn.OutputWeights[i])
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot32(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = nn.Activation(sum)
		}
	}

	return nn.OutputActivations
}

func (nn *FeedForward32) UpdateWithNoise(inputs []float32, noise [][]float32) []float32 {
	if len(inputs) != nn.NInputs-1 {
		log.Fatal("Error: wrong number of inputs")
	}

	if nn.Regression {
		for i := 0; i < nn.NInputs-1; i++ {
			nn.InputActivations[i] = inputs[i] + noise[0][i]
		}
	} else {
		for i := 0; i < nn.NInputs-1; i++ {
			nn.InputActivations[i] = normalize32(inputs[i] + noise[0][i])
		}
	}

	for i := 0; i < nn.NHiddens-1; i++ {
		sum := dot32(nn.InputActivations, nn.InputWeights[i])

		// compute contexts sum
		for k := 0; k < len(nn.Contexts); k++ {
			for j := 0; j < nn.NHiddens-1; j++ {
				sum += nn.Contexts[k][j]
			}
		}

		nn.HiddenActivations[i] = normalize32(nn.Activation(sum) + noise[1][i])
	}

	// update the contexts
	if len(nn.Contexts) > 0 {
		for i := len(nn.Contexts) - 1; i > 0; i-- {
			nn.Contexts[i] = nn.Contexts[i-1]
		}
		nn.Contexts[0] = nn.HiddenActivations
	}

	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot32(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = sum + noise[2][i]
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot32(nn.HiddenActivations, nn.OutputWeights[i])

			nn.OutputActivations[i] = normalize32(nn.Activation(sum) + noise[2][i])
		}
	}

	return nn.OutputActivations
}

/*
The BackPropagate method is used, when training the Neural Network,
to back propagate the errors from network activation.
*/
func (nn *FeedForward32) BackPropagate(targets []float32, lRate, mFactor float32) float32 {
	if len(targets) != nn.NOutputs {
		log.Fatal("Error: wrong number of target values")
	}

	outputDeltas := vector32(nn.NOutputs, 0.0)
	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			outputDeltas[i] = (targets[i] - nn.OutputActivations[i])
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			outputDeltas[i] = nn.DActivation(nn.OutputActivations[i]) * (targets[i] - nn.OutputActivations[i])
		}
	}

	hiddenDeltas := vector32(nn.NHiddens, 0.0)
	for i := 0; i < nn.NHiddens; i++ {
		var e float32

		for j := 0; j < nn.NOutputs; j++ {
			e += outputDeltas[j] * nn.OutputWeights[j][i]
		}

		hiddenDeltas[i] = nn.DActivation(nn.HiddenActivations[i]) * e
	}

	change := make([]float32, nn.NOutputs)
	for i := 0; i < nn.NHiddens; i++ {
		copy(change, outputDeltas)
		scal32(nn.HiddenActivations[i], change)
		scal32(mFactor, nn.OutputChanges[i])
		axpy32(lRate, change, nn.OutputChanges[i])
		for j := 0; j < nn.NOutputs; j++ {
			nn.OutputWeights[j][i] = nn.OutputWeights[j][i] + nn.OutputChanges[i][j]
		}
		copy(nn.OutputChanges[i], change)
	}

	change = make([]float32, nn.NHiddens)
	for i := 0; i < nn.NInputs; i++ {
		copy(change, hiddenDeltas)
		scal32(nn.InputActivations[i], change)
		scal32(mFactor, nn.InputChanges[i])
		axpy32(lRate, change, nn.InputChanges[i])
		for j := 0; j < nn.NHiddens; j++ {
			nn.InputWeights[j][i] = nn.InputWeights[j][i] + nn.InputChanges[i][j]
		}
		copy(nn.InputChanges[i], change)
	}

	var e float32

	for i := 0; i < len(targets); i++ {
		e += float32(math.Pow(float64(targets[i]-nn.OutputActivations[i]), 2))
	}

	return e
}

/*
This method is used to train the Network, it will run the training operation for 'iterations' times
and return the computed errors when training.
*/
func (nn *FeedForward32) Train(patterns [][][]float32, iterations int, lRate, mFactor float32, debug bool) []float32 {
	config := func(context *Context32) *Context32 {
		context.Iterations = iterations
		context.LRate = lRate
		context.MFactor = mFactor
		context.Debug = debug
		return context
	}
	return nn.TrainWithConfig(patterns, config)
}

func (nn *FeedForward32) TrainWithConfig(patterns [][][]float32, config Config32) []float32 {
	hidden, output := nn.Activation, nn.Activation

	//http://iamtrask.github.io/2015/07/28/dropout/
	if nn.Dropout != 0 {
		hidden = func(x float32) float32 {
			x = nn.Activation(x)
			if rand.Float32() > 1-nn.Dropout {
				x = 0
			} else {
				x *= 1 / (1 - nn.Dropout)
			}
			return x
		}
	}

	if nn.Regression {
		output = func(x float32) float32 {
			return x
		}
	}

	context := config(
		&Context32{
			Iterations:  10,
			LRate:       0.6,
			MFactor:     0.4,
			Debug:       false,
			Activations: []Activation32{hidden, output},
		},
	)

	errors := make([]float32, context.Iterations)

	for i := 0; i < context.Iterations; i++ {
		var e float32
		var n int
		for _, p := range patterns {
			nn.update(p[0], context)

			tmp := nn.BackPropagate(p[1], context.LRate, context.MFactor)
			e += tmp
			n += len(p[1])
		}

		errors[i] = e / float32(n)

		if context.Debug && i%1000 == 0 {
			fmt.Println(i, e)
		}
	}

	return errors
}

func (nn *FeedForward32) Test(patterns [][][]float32) {
	for _, p := range patterns {
		fmt.Println(p[0], "->", nn.Update(p[0]), " : ", p[1])
	}
}
