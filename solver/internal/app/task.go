package app

import (
	"context"
	"math"
	"math/rand"

	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/pkg/utils"
)

type Task struct {
	ID    int64
	State control.State
}

func taskFromPoint(id int64, x, y, z float64, base control.State) Task {
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
					task := taskFromPoint(id, x, y, z, base)

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
		maxRoll  = math.Pi / 18
		maxPitch = math.Pi / 6
		maxYaw   = math.Pi / 18

		minSpeed = 8.0
		maxSpeed = 12.0
	)

	controls := make([]control.Control, numIntervals)
	for i := range controls {
		controls[i] = control.Control{
			rnd.Float64()*2*maxRoll - maxRoll,
			rnd.Float64()*2*maxPitch - maxPitch,
			rnd.Float64()*2*maxYaw - maxYaw,
			rnd.Float64()*(maxSpeed-minSpeed) + minSpeed,
		}
	}

	return controls
}
