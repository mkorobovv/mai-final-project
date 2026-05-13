package trajectoryrepository

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/mkorobovv/mai-final-project/solver/internal/control"
)

func (repo *TrajectoryRepository) SaveStates(ctx context.Context, rows []control.Trajectory) (err error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	if len(rows) == 0 {
		return nil
	}

	batch := &pgx.Batch{}

	const query = `INSERT INTO trajectory_states (trajectory_id, position_id, state, control) VALUES ($1, $2, $3, $4)`

	for _, row := range rows {
		batch.Queue(query, row.TrajectoryID, row.PositionID, row.State, row.Control)
	}

	results := repo.pool.SendBatch(ctx, batch)
	defer func(results pgx.BatchResults) {
		errC := results.Close()
		if errC != nil {
			err = errors.Join(err, errC)
		}
	}(results)

	for i := 0; i < len(rows); i++ {
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

	const query = `INSERT INTO trajectories DEFAULT VALUES RETURNING trajectory_id`

	err = repo.pool.QueryRow(ctx, query).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("failed to execute statement: %w", err)
	}

	return id, nil
}
