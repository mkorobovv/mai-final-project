package control

// Output contains the best control sequence and its score produced by the optimizer.
type Output struct {
	BestControls []Control
	BestScore    float64
}
