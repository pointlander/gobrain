package gobrain

import (
	"math"
	"math/rand"
)

func random(a, b float64) float64 {
	return (b-a)*rand.Float64() + a
}

func matrix(I, J int) [][]float64 {
	m := make([][]float64, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float64, J)
	}
	return m
}

func vector(I int, fill float64) []float64 {
	v := make([]float64, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func normalize(a float64) float64 {
	if a > 1 {
		return 1
	} else if a < 0 {
		return 0
	}
	return a
}

func random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}

func matrix32(I, J int) [][]float32 {
	m := make([][]float32, I)
	for i := 0; i < I; i++ {
		m[i] = make([]float32, J)
	}
	return m
}

func vector32(I int, fill float32) []float32 {
	v := make([]float32, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func sigmoid32(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func dsigmoid32(y float32) float32 {
	return y * (1 - y)
}

func normalize32(a float32) float32 {
	if a > 1 {
		return 1
	} else if a < 0 {
		return 0
	}
	return a
}
