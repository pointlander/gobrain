// +build 386 arm

package gobrain

func dot64(X, Y []float64) float64 {
	var sum float64
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

func dot32(X, Y []float32) float32 {
	var sum float32
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

func scal64(alpha float64, X []float64) {
	for i, x := range X {
		X[i] = alpha * x
	}
}

func scal32(alpha float32, X []float32) {
	for i, x := range X {
		X[i] = alpha * x
	}
}

func axpy64(alpha float64, X []float64, Y []float64) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}

func axpy32(alpha float32, X []float32, Y []float32) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}
