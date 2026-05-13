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

func exitWithError(logger *slog.Logger, message string, err error) {
	logger.Error(message, slog.String("error", err.Error()))
	os.Exit(1)
}

func main() {
	ctx := context.Background()
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	cfg := app.DefaultConfig()
	if err := cfg.Validate(); err != nil {
		exitWithError(logger, "invalid generator config", err)
	}

	pgpool, err := postgres.New(ctx, cfg.Database)
	if err != nil {
		exitWithError(logger, "failed to initialize postgres", err)
	}
	defer pgpool.Close()

	scoreRepository := scorerepository.New(pgpool)
	trajectoryRepository := trajectoryrepository.New(pgpool)

	application := app.New(logger, cfg, scoreRepository, trajectoryRepository)
	if err := application.Start(ctx); err != nil {
		exitWithError(logger, "generator stopped with error", err)
	}
}
