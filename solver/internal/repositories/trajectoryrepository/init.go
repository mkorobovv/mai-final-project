package trajectoryrepository

import "github.com/jackc/pgx/v5/pgxpool"

type TrajectoryRepository struct {
	pool *pgxpool.Pool
}

func New(pool *pgxpool.Pool) *TrajectoryRepository {
	return &TrajectoryRepository{
		pool: pool,
	}
}
