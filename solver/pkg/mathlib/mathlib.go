package mathlib

import "math"

// Heaviside returns the value of the Heaviside step function.
func Heaviside(value float64) float64 {
	switch {
	case value > 0:
		return 1
	case value == 0:
		return 0.5
	default:
		return 0
	}
}

// Clamp limits value to the provided range.
func Clamp(value, min, max float64) float64 {
	switch {
	case value < min:
		return min
	case value > max:
		return max
	default:
		return value
	}
}

// EuclideanDistance computes Euclidean distance between two 6-D points.
func EuclideanDistance(current, target [6]float64) float64 {
	var distance float64

	for i := 0; i < len(current); i++ {
		distance += math.Pow(current[i]-target[i], 2)
	}

	return math.Sqrt(distance)
}
