package control

// Score represents optimization objective value for a trajectory.
type Score struct {
	ScoreID      int64
	TrajectoryID int64
	Score        float64
}
