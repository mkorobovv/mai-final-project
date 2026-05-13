package app

import (
	"fmt"

	"github.com/mkorobovv/mai-final-project/solver/internal/constraints"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/internal/infrastructure/postgres"
)

type InputConfig struct {
	NumIntervals       int
	NumIterations      int
	InitialTemperature float64
	Time               float64
	Cylinders          []constraints.Cylinder
	CylinderPenalty    float64
	Windows            []constraints.Window
	WindowPenalty      float64
	TerminalState      control.State
	TerminalPenalty    float64
	StepSize           float64
}

type TaskConfig struct {
	BaseState control.State
	Radius    float64
	Steps     int
}

type Config struct {
	Database postgres.Config
	Input    InputConfig
	Tasks    TaskConfig
}

func (c Config) Validate() error {
	switch {
	case c.Input.NumIntervals <= 0:
		return fmt.Errorf("num intervals must be positive")
	case c.Input.NumIterations <= 0:
		return fmt.Errorf("num iterations must be positive")
	case c.Input.Time <= 0:
		return fmt.Errorf("time must be positive")
	case c.Input.StepSize <= 0:
		return fmt.Errorf("step size must be positive")
	case c.Tasks.Steps <= 0:
		return fmt.Errorf("task steps must be positive")
	case c.Tasks.Radius < 0:
		return fmt.Errorf("task radius must be non-negative")
	default:
		return nil
	}
}

func DefaultConfig() Config {
	return Config{
		Database: postgres.Config{
			Host:     "localhost",
			Port:     5432,
			User:     "postgres",
			Password: "postgres",
			Database: "trajectory",
		},
		Input: InputConfig{
			NumIntervals:       15,
			NumIterations:      200_000,
			InitialTemperature: 200,
			Time:               5.6,
			Cylinders: []constraints.Cylinder{
				{
					Coordinates: [3]float64{1.5, 0.0, 2.5},
					Radius:      2.5,
				},
				{
					Coordinates: [3]float64{6.5, 0.0, 7.5},
					Radius:      2.5,
				},
			},
			CylinderPenalty: 0.9,
			Windows: []constraints.Window{
				{
					Coordinates: [3]float64{4.0, 0.0, 5.0},
					Radius:      0.5,
				},
			},
			WindowPenalty:   1.6,
			TerminalState:   control.State{5, 5, 10, 0, 0, 0},
			TerminalPenalty: 0.9,
			StepSize:        0.01,
		},
		Tasks: TaskConfig{
			BaseState: control.State{0, 0, 0, 0, 0, 0},
			Radius:    0.45,
			Steps:     5,
		},
	}
}
