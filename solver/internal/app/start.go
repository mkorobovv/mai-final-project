package app

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"runtime"
	"time"

	"github.com/mkorobovv/mai-final-project/solver/internal/annealing"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
	"github.com/mkorobovv/mai-final-project/solver/internal/dynamics"
	"github.com/mkorobovv/mai-final-project/solver/internal/functional"
	"golang.org/x/sync/errgroup"
)

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
	workerCount := runtime.NumCPU()
	if workerCount < 1 {
		workerCount = 1
	}

	app.logger.Info("starting optimizations", slog.Int("workers", workerCount))

	tasks := neighborhoodGenerator(ctx, app.cfg.Tasks)

	group, groupCtx := errgroup.WithContext(ctx)

	for workerID := 0; workerID < workerCount; workerID++ {
		seed := time.Now().UnixNano() + int64(workerID)*1_000_000

		group.Go(func() error {
			rnd := rand.New(rand.NewSource(seed))

			for task := range tasks {
				if err := app.processTask(groupCtx, task, rnd); err != nil {
					return fmt.Errorf("process task %d: %w", task.ID, err)
				}
			}

			return nil
		})
	}

	return group.Wait()
}

func (app *App) processTask(ctx context.Context, task Task, rnd *rand.Rand) error {
	input := app.cfg.Input
	initialState := task.State
	initialControls := generateControls(input.NumIntervals, rnd)

	trajectoryID, err := app.trajectoryRepository.CreateTrajectory(ctx)
	if err != nil {
		return err
	}

	costSvc := functional.New(
		functional.Config{
			StepSize:        input.Time / float64(input.NumIntervals),
			TerminalState:   input.TerminalState,
			TerminalPenalty: input.TerminalPenalty,
			Cylinders:       input.Cylinders,
			CylinderPenalty: input.CylinderPenalty,
			Windows:         input.Windows,
			WindowPenalty:   input.WindowPenalty,
		},
		dynamics.Model,
	)

	optimizer := annealing.New(
		annealing.Config{
			NumIterations:      input.NumIterations,
			StepSize:           input.StepSize,
			InitialTemperature: input.InitialTemperature,
			InitialState:       initialState,
			InitialControls:    initialControls,
			Rnd:                rnd,
		},
		costSvc,
	)

	optimized := optimizer.Optimize()
	trajectory := costSvc.Trajectory(initialState, optimized.BestControls)
	states := buildTrajectoryRows(trajectoryID, trajectory, optimized.BestControls)

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

func buildTrajectoryRows(trajectoryID int64, states []control.State, controls []control.Control) []control.Trajectory {
	rows := make([]control.Trajectory, 0, len(controls))

	for idx := range controls {
		rows = append(rows, control.Trajectory{
			TrajectoryID: trajectoryID,
			PositionID:   int64(idx),
			State:        states[idx].ToSlice(),
			Control:      controls[idx].ToSlice(),
		})
	}

	return rows
}
