package gobrain

import (
	"math/rand"
	"testing"
)

func ExampleSimpleFeedForward() {
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	// instantiate the Feed Forward
	ff := &FeedForward{}

	// initialize the Neural Network;
	// the networks structure will contain:
	// 2 inputs, 2 hidden nodes and 1 output.
	ff.Init(2, 2, 1)

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	ff.Train(patterns, 1000, 0.6, 0.4, false)

	// testing the network
	ff.Test(patterns)

	// predicting a value
	inputs := []float64{1, 1}
	ff.Update(inputs)

	// Output:
	// [0 0] -> [0.05750394570844577]  :  [0]
	// [0 1] -> [0.9301006350712095]  :  [1]
	// [1 0] -> [0.9278099662272834]  :  [1]
	// [1 1] -> [0.09740879532462136]  :  [0]
}

func ExampleSimpleFeedForwardDropOut() {
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	// instantiate the Feed Forward
	ff := &FeedForward{}
	// set the dropout probability
	ff.Dropout = .2
	// initialize the Neural Network;
	// the networks structure will contain:
	ff.Init(2, 4, 1)
	// 2 inputs, 4 hidden nodes and 1 output.
	// more than 3 hidden nodes are needed for dropout

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	ff.Train(patterns, 60000, 0.6, 0.4, false)

	// testing the network
	ff.Test(patterns)

	// predicting a value
	inputs := []float64{1, 1}
	ff.Update(inputs)

	// Output:
	// [0 0] -> [0.010904292959260731]  :  [0]
	// [0 1] -> [0.9972511065063475]  :  [1]
	// [1 0] -> [0.9972423459545571]  :  [1]
	// [1 1] -> [0.0972657515200182]  :  [0]
}

func BenchmarkFeedForward(b *testing.B) {
	rand.Seed(0)
	patterns := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	for n := 0; n < b.N; n++ {
		ff := &FeedForward{}
		ff.Init(2, 2, 1)
		ff.Train(patterns, 1000, 0.6, 0.4, false)
	}
}

func ExampleSimpleFeedForward32() {
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	patterns := [][][]float32{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	// instantiate the Feed Forward
	ff := &FeedForward32{}

	// initialize the Neural Network;
	// the networks structure will contain:
	// 2 inputs, 2 hidden nodes and 1 output.
	ff.Init(2, 2, 1)

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	ff.Train(patterns, 1000, 0.6, 0.4, false)

	// testing the network
	ff.Test(patterns)

	// predicting a value
	inputs := []float32{1, 1}
	ff.Update(inputs)

	// Output:
	// [0 0] -> [0.05750396]  :  [0]
	// [0 1] -> [0.9301007]  :  [1]
	// [1 0] -> [0.92781013]  :  [1]
	// [1 1] -> [0.09740849]  :  [0]
}

func ExampleSimpleFeedForwardDropOut32() {
	// set the random seed to 0
	rand.Seed(0)

	// create the XOR representation patter to train the network
	patterns := [][][]float32{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	// instantiate the Feed Forward
	ff := &FeedForward32{}
	// set the dropout probability
	ff.Dropout = .2
	// initialize the Neural Network;
	// the networks structure will contain:
	ff.Init(2, 4, 1)
	// 2 inputs, 4 hidden nodes and 1 output.
	// more than 3 hidden nodes are needed for dropout

	// train the network using the XOR patterns
	// the training will run for 1000 epochs
	// the learning rate is set to 0.6 and the momentum factor to 0.4
	// use true in the last parameter to receive reports about the learning error
	ff.Train(patterns, 60000, 0.6, 0.4, false)

	// testing the network
	ff.Test(patterns)

	// predicting a value
	inputs := []float32{1, 1}
	ff.Update(inputs)

	// Output:
	// [0 0] -> [0.010906684]  :  [0]
	// [0 1] -> [0.9972458]  :  [1]
	// [1 0] -> [0.99723935]  :  [1]
	// [1 1] -> [0.0972718]  :  [0]
}

func BenchmarkFeedForward32(b *testing.B) {
	rand.Seed(0)
	patterns := [][][]float32{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}
	for n := 0; n < b.N; n++ {
		ff := &FeedForward32{}
		ff.Init(2, 2, 1)
		ff.Train(patterns, 1000, 0.6, 0.4, false)
	}
}
