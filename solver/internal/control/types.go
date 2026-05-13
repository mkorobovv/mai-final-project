package control

// State represents the 6-dimensional quadrotor state vector.
type State [6]float64

// ToSlice returns a copy of the state as a slice for convenient serialization.
func (s State) ToSlice() []float64 {
	values := make([]float64, len(s))
	copy(values, s[:])

	return values
}

// Control represents the 4-dimensional control input (phi, theta, psi, thrust).
type Control [4]float64

// ToSlice returns a copy of the control as a slice for convenient serialization.
func (c Control) ToSlice() []float64 {
	values := make([]float64, len(c))
	copy(values, c[:])

	return values
}
