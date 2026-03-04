package scorerepository

import "github.com/jackc/pgx/v5/pgxpool"

type ScoreRepository struct {
	pool *pgxpool.Pool
}

func New(pool *pgxpool.Pool) *ScoreRepository {
	return &ScoreRepository{
		pool: pool,
	}
}
