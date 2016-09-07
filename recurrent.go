package gobrain

import "log"

type RNN32 struct {
	// Number of input, hidden and output nodes
	inputs, NInputs, NHiddens, NOutputs int
	// Whether it is regression or not
	Regression bool
	// Activations for nodes
	InputActivations, HiddenActivations []float32
	// Weights
	InputWeights [][]float32
}

func (nn *RNN32) Init(inputs, hiddens, outputs int) {
	nn.inputs = inputs
	nn.NInputs = inputs + hiddens + 1
	nn.NHiddens = hiddens + outputs
	nn.NOutputs = outputs

	nn.InputActivations = vector32(nn.NInputs, 1.0)
	nn.HiddenActivations = vector32(nn.NHiddens, 1.0)

	nn.InputWeights = matrix32(nn.NHiddens, nn.NInputs)
}

func (nn *RNN32) SetWeights(weights []float32) {
	w := 0
	for i := 0; i < nn.NHiddens; i++ {
		for j := 0; j < nn.NInputs; j++ {
			nn.InputWeights[i][j] = weights[w]
			w++
		}
	}
}

func (nn *RNN32) Reset() {
	for i := nn.NOutputs; i < nn.NHiddens; i++ {
		nn.HiddenActivations[i] = 0
	}
}

func (nn *RNN32) Update(inputs []float32) []float32 {
	if len(inputs) != nn.inputs {
		log.Fatal("Error: wrong number of inputs")
	}

	copy(nn.InputActivations, inputs)
	copy(nn.InputActivations[nn.inputs:], nn.HiddenActivations[nn.NOutputs:nn.NHiddens])

	if nn.Regression {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot32(nn.InputActivations, nn.InputWeights[i])
			nn.HiddenActivations[i] = sum
		}
	} else {
		for i := 0; i < nn.NOutputs; i++ {
			sum := dot32(nn.InputActivations, nn.InputWeights[i])
			nn.HiddenActivations[i] = tanh32(sum)
		}
	}
	for i := nn.NOutputs; i < nn.NHiddens; i++ {
		sum := dot32(nn.InputActivations, nn.InputWeights[i])
		nn.HiddenActivations[i] = tanh32(sum)
	}

	return nn.HiddenActivations[:nn.NOutputs]
}
