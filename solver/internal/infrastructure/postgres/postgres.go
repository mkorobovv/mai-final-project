package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/mkorobovv/mai-final-project/solver/migrations"
	"github.com/pressly/goose/v3"
)

type Config struct {
	Host     string
	Port     int
	User     string
	Password string
	Database string
}

func (c Config) ConnString() string {
	return fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=disable",
		c.User, c.Password, c.Host, c.Port, c.Database)
}

func New(ctx context.Context, config Config) (*pgxpool.Pool, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	connString := config.ConnString()

	err := runMigrations(connString)
	if err != nil {
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	pool, err := pgxpool.New(ctx, connString)
	if err != nil {
		return nil, fmt.Errorf("failed to create pgx pool: %w", err)
	}

	err = pool.Ping(ctx)
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return pool, nil
}

func runMigrations(connString string) (err error) {
	db, err := sql.Open("pgx", connString)
	if err != nil {
		return err
	}
	defer func() {
		if errC := db.Close(); errC != nil {
			err = errors.Join(err, errC)
		}
	}()

	goose.SetBaseFS(migrations.FS)

	if err = goose.SetDialect("postgres"); err != nil {
		return err
	}

	if err = goose.Up(db, "."); err != nil {
		return err
	}

	return nil
}
