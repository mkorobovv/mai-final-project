package dynamics

import (
	"math"

	"github.com/mkorobovv/mai-final-project/solver/internal/control"
)

const EarthGravity = 9.81

// Model describes the quadrotor translational dynamics.
// state: [x1, x2, x3, v1, v2, v3]
// control: [phi, theta, psi, thrust]
func Model(state control.State, controlInput control.Control) control.State {
	phi, theta, psi, thrust := controlInput[0], controlInput[1], controlInput[2], controlInput[3]

	return control.State{
		state[3],
		state[4],
		state[5],
		(math.Cos(psi)*math.Sin(theta)*math.Cos(phi) + math.Sin(psi)*math.Sin(phi)) * thrust,
		(math.Sin(psi)*math.Sin(theta)*math.Cos(phi) - math.Cos(psi)*math.Sin(phi)) * thrust,
		thrust*math.Cos(theta)*math.Cos(phi) - EarthGravity,
	}
}

// Handler is a convenience alias for dependency injection.
type Handler func(state control.State, control control.Control) control.State
