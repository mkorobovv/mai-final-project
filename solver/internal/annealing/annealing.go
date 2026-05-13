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

func (a *Annealing) Optimize() control.Output {
	bestControls := cloneControls(a.config.InitialControls)
	bestScore := a.costService.Cost(a.config.InitialState, bestControls)

	currentControls := cloneControls(bestControls)
	currentScore := bestScore

	for i := 0; i < a.config.NumIterations; i++ {
		temperature := a.temperature(i)

		candidateControls := a.neighborControls(currentControls)
		candidateScore := a.costService.Cost(a.config.InitialState, candidateControls)

		accept := candidateScore < currentScore ||
			a.config.Rnd.Float64() < math.Exp((currentScore-candidateScore)/temperature)

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

func (a *Annealing) temperature(iteration int) float64 {
	temperature := a.config.InitialTemperature / (1 + 0.01*float64(iteration))
	if temperature < 1e-9 {
		return 1e-9
	}

	return temperature
}

func (a *Annealing) neighborControls(controls []control.Control) []control.Control {
	neighbor := make([]control.Control, len(controls))

	for i := range controls {
		for j := 0; j < 4; j++ {
			neighbor[i][j] = controls[i][j] + a.config.Rnd.NormFloat64()*a.config.StepSize
		}

		neighbor[i][0] = mathlib.Clamp(neighbor[i][0], -math.Pi/12, math.Pi/12)
		neighbor[i][1] = mathlib.Clamp(neighbor[i][1], -math.Pi, math.Pi)
		neighbor[i][2] = mathlib.Clamp(neighbor[i][2], -math.Pi/12, math.Pi/12)
		neighbor[i][3] = mathlib.Clamp(neighbor[i][3], 0, 12)
	}

	return neighbor
}

func cloneControls(controls []control.Control) []control.Control {
	cloned := make([]control.Control, len(controls))
	copy(cloned, controls)

	return cloned
}
