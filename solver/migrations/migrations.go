package migrations

import "embed"

// FS exports the embedded migrations filesystem
//
//go:embed *.sql
var FS embed.FS
