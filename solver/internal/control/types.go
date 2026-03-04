package control

// State represents the 6-dimensional quadrotor state vector.
type State [6]float64

// ToSlice returns a copy of the state as a slice for convenient serialization.
func (s State) ToSlice() []float64 {
	st := make([]float64, len(s))
	for i := range s {
		st[i] = s[i]
	}
	return st
}

// Control represents the 4-dimensional control input (phi, theta, psi, thrust).
type Control [4]float64

// ToSlice returns a copy of the control as a slice for convenient serialization.
func (c Control) ToSlice() []float64 {
	ct := make([]float64, len(c))
	for i := range c {
		ct[i] = c[i]
	}
	return ct
}
