package constraints

import "math"

// Window defines a circular opening that the trajectory should pass through.
type Window struct {
	Coordinates [3]float64
	Radius      float64
}

// Distance returns signed distance to the window boundary (negative when inside).
func (w Window) Distance(x1, x3 float64) float64 {
	return math.Hypot(x1-w.Coordinates[0], x3-w.Coordinates[2]) - w.Radius
}

// Penalty grows quadratically outside the window to encourage proximity.
func (w Window) Penalty(value float64) float64 {
	if value <= 0 {
		return 0
	}
	return value * value
}
