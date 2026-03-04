package annealing

import (
	"math"
	"math/rand"

	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/pkg/mathlib"
)

type Config struct {
	NumIterations      int
	StepSize           float64
	InitialTemperature float64
	InitialState       control.State
	InitialControls    []control.Control
	Rnd                *rand.Rand
}

type costService interface {
	Cost(state control.State, controls []control.Control) float64
}

type Annealing struct {
	config      Config
	costService costService
}

func New(config Config, costService costService) *Annealing {
	return &Annealing{config: config, costService: costService}
}

// Optimize runs the simulated annealing search.
func (a *Annealing) Optimize() control.Output {
	bestControls := a.config.InitialControls
	bestScore := a.costService.Cost(a.config.InitialState, bestControls)

	currentControls := bestControls
	currentScore := bestScore

	temperature := func(i int) float64 {
		return a.config.InitialTemperature / (1 + 0.01*float64(i))
	}

	for i := 0; i < a.config.NumIterations; i++ {
		t := temperature(i)

		candidateControls := a.GetNeighbor(currentControls, a.config.StepSize)
		candidateScore := a.costService.Cost(a.config.InitialState, candidateControls)

		accept := candidateScore < currentScore ||
			a.config.Rnd.Float64() < math.Exp((currentScore-candidateScore)/t)

		if accept {
			currentControls = candidateControls
			currentScore = candidateScore

			if candidateScore < bestScore {
				bestControls = candidateControls
				bestScore = candidateScore
			}
		}
	}

	return control.Output{
		BestControls: bestControls,
		BestScore:    bestScore,
	}
}

// GetNeighbor generates a neighboring control sequence.
func (a *Annealing) GetNeighbor(controls []control.Control, stepSize float64) []control.Control {
	neighbor := make([]control.Control, len(controls))

	for i := range controls {
		for j := 0; j < 4; j++ {
			neighbor[i][j] = controls[i][j] + a.config.Rnd.NormFloat64()*stepSize
		}

		neighbor[i][0] = mathlib.Clamp(neighbor[i][0], -math.Pi/12, math.Pi/12)
		neighbor[i][1] = mathlib.Clamp(neighbor[i][1], -math.Pi, math.Pi)
		neighbor[i][2] = mathlib.Clamp(neighbor[i][2], -math.Pi/12, math.Pi/12)
		neighbor[i][3] = mathlib.Clamp(neighbor[i][3], 0, 12)
	}

	return neighbor
}
