package trajectoryrepository

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
)

func (repo *TrajectoryRepository) SaveStates(ctx context.Context, trajectories []control.Trajectory) (err error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	batch := &pgx.Batch{}

	const query = `INSERT INTO trajectory_states (trajectory_id, position_id, state, control) VALUES ($1, $2, $3, $4)`

	for _, trajectory := range trajectories {
		batch.Queue(query, trajectory.TrajectoryID, trajectory.PositionID, trajectory.State, trajectory.Control)
	}

	results := repo.pool.SendBatch(ctx, batch)
	defer func(results pgx.BatchResults) {
		errC := results.Close()
		if errC != nil {
			err = errors.Join(err, errC)
		}
	}(results)

	for i := 0; i < len(trajectories); i++ {
		_, err = results.Exec()
		if err != nil {
			return fmt.Errorf("failed to execute statement %d: %w", i, err)
		}
	}

	return nil
}

func (repo *TrajectoryRepository) CreateTrajectory(ctx context.Context) (id int64, err error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	query := `INSERT INTO trajectories DEFAULT VALUES RETURNING trajectory_id`

	err = repo.pool.QueryRow(ctx, query).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to execute statement: %w", err)
	}

	return id, nil
}
