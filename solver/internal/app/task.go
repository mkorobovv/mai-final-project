package app

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/mkorobovv/mai-final-project/solver/internal/constraints"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/pkg/utils"
)

type TaskConfig struct {
	BaseState control.State
	Radius    float64
	Steps     int
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
	}

	return nil
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
		Input: struct {
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
			InitialState       control.State
			InitialControls    []control.Control
		}{
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

type Task struct {
	ID    int64
	State control.State
}

func newTask(id int64, x, y, z float64, base control.State) Task {
	return Task{
		ID: id,
		State: control.State{
			x, y, z,
			base[3], base[4], base[5],
		},
	}
}

func linspace(base, radius float64, n int) []float64 {
	if n <= 1 {
		return []float64{base}
	}

	minVal := base - radius
	maxVal := base + radius
	step := (maxVal - minVal) / float64(n-1)

	points := make([]float64, n)

	for i := range points {
		points[i] = minVal + float64(i)*step
	}

	return points
}

func neighborhoodGenerator(ctx context.Context, cfg TaskConfig) <-chan Task {
	out := make(chan Task)

	go func() {
		defer close(out)

		var id int64 = 1

		xPoints := linspace(cfg.BaseState[0], cfg.Radius, cfg.Steps)
		yPoints := linspace(cfg.BaseState[1], cfg.Radius, cfg.Steps)
		zPoints := linspace(cfg.BaseState[2], cfg.Radius, cfg.Steps)

		base := cfg.BaseState

		for _, x := range xPoints {
			for _, y := range yPoints {
				for _, z := range zPoints {
					task := newTask(id, x, y, z, base)

					if ok := utils.SendOrDone(ctx, out, task); !ok {
						return
					}

					id++
				}
			}
		}
	}()

	return out
}

func generateControls(numIntervals int, rnd *rand.Rand) []control.Control {
	const (
		rollRange  = math.Pi / 9
		pitchRange = math.Pi / 3
		yawRange   = math.Pi / 9

		speedMin = 8.0
		speedMax = 12.0
	)

	half := func(x float64) float64 { return x / 2 }

	controls := make([]control.Control, numIntervals)
	for i := range controls {
		controls[i] = control.Control{
			rnd.Float64()*rollRange - half(rollRange),
			rnd.Float64()*pitchRange - half(pitchRange),
			rnd.Float64()*yawRange - half(yawRange),
			rnd.Float64()*(speedMax-speedMin) + speedMin,
		}
	}
	return controls
}
