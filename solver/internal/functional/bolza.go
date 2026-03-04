package functional

import (
	"math"

	"github.com/mkorobovv/mai-final-project/solver/internal/constraints"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/internal/dynamics"
	"github.com/mkorobovv/mai-final-project/solver/pkg/mathlib"
)

// Config stores parameters required to evaluate the Bolza-type cost functional.
type Config struct {
	NumIntervals int
	RK45Step     float64

	TerminalState   control.State
	TerminalPenalty float64

	Cylinders       []constraints.Cylinder
	CylinderPenalty float64

	Windows       []constraints.Window
	WindowPenalty float64
}

// Service evaluates costs and trajectories for a given dynamics model.
type Service struct {
	cfg     Config
	handler dynamics.Handler
}

// New creates a Service bound to a specific dynamics function (e.g., dynamics.Model).
func New(cfg Config, handler dynamics.Handler) *Service {
	return &Service{cfg: cfg, handler: handler}
}

// Cost computes the Bolza functional value for the provided initial state and control sequence.
func (s *Service) Cost(initialState control.State, controls []control.Control) float64 {
	var (
		cylinderPenalty float64
		windowPenalty   float64
	)

	states := s.Trajectory(initialState, controls)

	for _, st := range states {
		for _, cylinder := range s.cfg.Cylinders {
			distance := cylinder.Distance(st[0], st[2])
			cylinderPenalty += s.cfg.CylinderPenalty * cylinder.Penalty(distance)
		}
	}

	for _, window := range s.cfg.Windows {
		minDistance := math.Inf(1)

		for _, st := range states {
			distance := window.Distance(st[0], st[2])
			if distance < minDistance {
				minDistance = distance
			}
		}

		windowPenalty += s.cfg.WindowPenalty * window.Penalty(minDistance)
	}

	terminalPenalty := s.cfg.TerminalPenalty * mathlib.EuclideanDistance(states[s.cfg.NumIntervals], s.cfg.TerminalState)

	return s.cfg.RK45Step*float64(s.cfg.NumIntervals) + terminalPenalty + cylinderPenalty + windowPenalty
}

// Trajectory integrates the system dynamics for the provided controls.
func (s *Service) Trajectory(initial control.State, controls []control.Control) []control.State {
	states := make([]control.State, s.cfg.NumIntervals+1)
	states[0] = initial

	for k := 0; k < s.cfg.NumIntervals; k++ {
		states[k+1] = s.RK45Step(states[k], controls[k])
	}

	return states
}

// RK45Step integrates one step using classical 4th-order Runge-Kutta (fixed step size).
func (s *Service) RK45Step(state control.State, controlInput control.Control) control.State {
	var newState control.State

	k1 := s.handler(state, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + 0.5*s.cfg.RK45Step*k1[i]
	}

	k2 := s.handler(newState, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + 0.5*s.cfg.RK45Step*k2[i]
	}

	k3 := s.handler(newState, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + s.cfg.RK45Step*k3[i]
	}

	k4 := s.handler(newState, controlInput)

	var finalState control.State

	for i := 0; i < len(state); i++ {
		finalState[i] = state[i] + (s.cfg.RK45Step/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
	}

	return finalState
}
