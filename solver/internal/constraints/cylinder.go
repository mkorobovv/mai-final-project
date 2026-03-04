package constraints

import (
	"math"

	"github.com/mkorobovv/mai-final-project/solver/pkg/mathlib"
)

// Cylinder describes a cylindrical obstacle in the x1-x3 plane.
type Cylinder struct {
	Coordinates [3]float64
	Radius      float64
}

// Distance returns signed distance from the trajectory point (x1, x3) to the cylinder surface.
// Positive value means the point is inside the cylinder and incurs penalty.
func (c Cylinder) Distance(x1, x3 float64) float64 {
	return c.Radius - math.Hypot(x1-c.Coordinates[0], x3-c.Coordinates[2])
}

// Penalty applies a Heaviside-based penalty to the signed distance.
func (c Cylinder) Penalty(value float64) float64 {
	return mathlib.Heaviside(value)
}
