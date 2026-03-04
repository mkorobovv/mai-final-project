package app

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"runtime"
	"time"

	"github.com/mkorobovv/mai-final-project/solver/internal/annealing"
	"github.com/mkorobovv/mai-final-project/solver/internal/constraints"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/internal/dynamics"
	"github.com/mkorobovv/mai-final-project/solver/internal/functional"
	"github.com/mkorobovv/mai-final-project/solver/internal/infrastructure/postgres"
	"golang.org/x/sync/errgroup"
)

type Config struct {
	Database postgres.Config
	Input    struct {
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
	}
	Tasks TaskConfig
}

type App struct {
	logger               *slog.Logger
	cfg                  Config
	scoreRepository      scoreRepository
	trajectoryRepository trajectoryRepository
}

type trajectoryRepository interface {
	CreateTrajectory(ctx context.Context) (id int64, err error)
	SaveStates(ctx context.Context, trajectories []control.Trajectory) error
}

type scoreRepository interface {
	SaveScore(ctx context.Context, score control.Score) error
}

func New(logger *slog.Logger, cfg Config, scoreRepository scoreRepository, trajectoryRepository trajectoryRepository) *App {
	return &App{
		logger:               logger,
		cfg:                  cfg,
		scoreRepository:      scoreRepository,
		trajectoryRepository: trajectoryRepository,
	}
}

func (app *App) Start(ctx context.Context) error {
	workers := runtime.NumCPU()
	if workers < 1 {
		workers = 1
	}

	app.logger.Info("Starting optimizations...", slog.Int("workers", workers))

	tasks := neighborhoodGenerator(ctx, app.cfg.Tasks)

	g, gCtx := errgroup.WithContext(ctx)

	for id := 0; id < workers; id++ {
		seed := time.Now().UnixNano() + int64(id)*1_000_000

		g.Go(func() error {
			rnd := rand.New(rand.NewSource(seed))

			for task := range tasks {
				if err := app.processTask(gCtx, task, rnd); err != nil {
					return fmt.Errorf("process task %d: %w", task.ID, err)
				}
			}

			return nil
		})
	}

	return g.Wait()
}

func (app *App) processTask(ctx context.Context, task Task, rnd *rand.Rand) error {
	controls := generateControls(app.cfg.Input.NumIntervals, rnd)

	inp := app.cfg.Input
	inp.InitialState = task.State

	trajectoryID, err := app.trajectoryRepository.CreateTrajectory(ctx)
	if err != nil {
		return err
	}

	costSvc := functional.New(
		functional.Config{
			NumIntervals:    inp.NumIntervals,
			RK45Step:        inp.Time / float64(inp.NumIntervals),
			TerminalState:   inp.TerminalState,
			TerminalPenalty: inp.TerminalPenalty,
			Cylinders:       inp.Cylinders,
			CylinderPenalty: inp.CylinderPenalty,
			Windows:         inp.Windows,
			WindowPenalty:   inp.WindowPenalty,
		},
		dynamics.Model,
	)

	annealingAlg := annealing.New(
		annealing.Config{
			NumIterations:      inp.NumIterations,
			StepSize:           inp.StepSize,
			InitialTemperature: inp.InitialTemperature,
			InitialState:       inp.InitialState,
			InitialControls:    controls,
			Rnd:                rnd,
		},
		costSvc,
	)

	optimized := annealingAlg.Optimize()
	trajectory := costSvc.Trajectory(inp.InitialState, optimized.BestControls)

	states := make([]control.Trajectory, 0, len(trajectory))
	for idx := range inp.NumIntervals {
		stateRow := control.Trajectory{
			TrajectoryID: trajectoryID,
			PositionID:   int64(idx),
			State:        trajectory[idx].ToSlice(),
			Control:      optimized.BestControls[idx].ToSlice(),
		}

		states = append(states, stateRow)
	}

	if err = app.trajectoryRepository.SaveStates(ctx, states); err != nil {
		app.logger.Error(err.Error(), slog.String("source", "save trajectory"), slog.Int64("task_id", task.ID))

		return err
	}

	if err = app.scoreRepository.SaveScore(ctx, control.Score{TrajectoryID: trajectoryID, Score: optimized.BestScore}); err != nil {
		app.logger.Error(err.Error(), slog.String("source", "save score"), slog.Int64("task_id", task.ID))

		return err
	}

	return nil
}
