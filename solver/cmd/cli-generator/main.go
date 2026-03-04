package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/mkorobovv/mai-final-project/solver/internal/app"
	"github.com/mkorobovv/mai-final-project/solver/internal/infrastructure/postgres"
	"github.com/mkorobovv/mai-final-project/solver/internal/repositories/scorerepository"
	"github.com/mkorobovv/mai-final-project/solver/internal/repositories/trajectoryrepository"
)

func main() {
	ctx := context.Background()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg := app.DefaultConfig()
	if err := cfg.Validate(); err != nil {
		logger.Error("invalid generator config", slog.String("error", err.Error()))
		os.Exit(1)
	}

	pgpool, err := postgres.New(ctx, cfg.Database)
	if err != nil {
		logger.Error("failed to initialize postgres", slog.String("error", err.Error()))
		os.Exit(1)
	}
	defer pgpool.Close()

	scoreRepository := scorerepository.New(pgpool)
	trajectoryRepository := trajectoryrepository.New(pgpool)

	a := app.New(logger, cfg, scoreRepository, trajectoryRepository)
	if err := a.Start(ctx); err != nil {
		logger.Error("generator stopped with error", slog.String("error", err.Error()))
		os.Exit(1)
	}
}
