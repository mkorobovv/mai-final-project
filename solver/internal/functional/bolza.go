package functional

import (
	"math"

	"github.com/mkorobovv/mai-final-project/solver/internal/constraints"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/internal/dynamics"
	"github.com/mkorobovv/mai-final-project/solver/pkg/mathlib"
)

type Config struct {
	StepSize float64

	TerminalState   control.State
	TerminalPenalty float64

	Cylinders       []constraints.Cylinder
	CylinderPenalty float64

	Windows       []constraints.Window
	WindowPenalty float64
}

type Service struct {
	cfg     Config
	handler dynamics.Handler
}

func New(cfg Config, handler dynamics.Handler) *Service {
	return &Service{cfg: cfg, handler: handler}
}

func (s *Service) Cost(initialState control.State, controls []control.Control) float64 {
	states := s.Trajectory(initialState, controls)
	terminalPenalty := s.cfg.TerminalPenalty * mathlib.EuclideanDistance(states[len(states)-1], s.cfg.TerminalState)

	return s.cfg.StepSize*float64(len(controls)) +
		terminalPenalty +
		s.cylinderPenalty(states) +
		s.windowPenalty(states)
}

func (s *Service) Trajectory(initial control.State, controls []control.Control) []control.State {
	states := make([]control.State, len(controls)+1)
	states[0] = initial

	for idx, controlInput := range controls {
		states[idx+1] = s.rk4Step(states[idx], controlInput)
	}

	return states
}

func (s *Service) cylinderPenalty(states []control.State) float64 {
	var penalty float64

	for _, state := range states {
		for _, cylinder := range s.cfg.Cylinders {
			penalty += s.cfg.CylinderPenalty * cylinder.Penalty(cylinder.Distance(state[0], state[2]))
		}
	}

	return penalty
}

func (s *Service) windowPenalty(states []control.State) float64 {
	var penalty float64

	for _, window := range s.cfg.Windows {
		minDistance := math.Inf(1)

		for _, state := range states {
			if distance := window.Distance(state[0], state[2]); distance < minDistance {
				minDistance = distance
			}
		}

		penalty += s.cfg.WindowPenalty * window.Penalty(minDistance)
	}

	return penalty
}

func (s *Service) rk4Step(state control.State, controlInput control.Control) control.State {
	var newState control.State

	k1 := s.handler(state, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + 0.5*s.cfg.StepSize*k1[i]
	}

	k2 := s.handler(newState, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + 0.5*s.cfg.StepSize*k2[i]
	}

	k3 := s.handler(newState, controlInput)

	for i := 0; i < len(state); i++ {
		newState[i] = state[i] + s.cfg.StepSize*k3[i]
	}

	k4 := s.handler(newState, controlInput)

	var finalState control.State

	for i := 0; i < len(state); i++ {
		finalState[i] = state[i] + (s.cfg.StepSize/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i])
	}

	return finalState
}
