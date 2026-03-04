package control

import "time"

// Trajectory holds a single state/control pair at a given position.
type Trajectory struct {
	TrajectoryID int64
	PositionID   int64
	State        []float64
	Control      []float64
	CreatedAt    time.Time
}
