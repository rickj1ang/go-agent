package agent

import (
	"context"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

func TestRegexBlocklistPolicy(t *testing.T) {
	patterns := []string{
		`(?i)\b(?:password|secret)\s*=\s*\w+`,
		`\b\d{3}-\d{2}-\d{4}\b`, // SSN-like
	}

	policy, err := NewRegexBlocklistPolicy(patterns)
	if err != nil {
		t.Fatalf("Failed to create policy: %v", err)
	}

	tests := []struct {
		name      string
		response  string
		wantError bool
	}{
		{
			name:      "Clean text",
			response:  "This is a perfectly safe response.",
			wantError: false,
		},
		{
			name:      "Password leak",
			response:  "Here is the config, password=secret123",
			wantError: true,
		},
		{
			name:      "SSN leak",
			response:  "My SSN is 123-45-6789.",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := policy.Validate(context.Background(), tt.response)
			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// mockSafetyModel implements models.Agent for testing
type mockSafetyModel struct {
	response string
	err      error
}

func (m *mockSafetyModel) Generate(ctx context.Context, prompt string) (any, error) {
	return m.response, m.err
}

func (m *mockSafetyModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.response, m.err
}

func (m *mockSafetyModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{
		Delta:    m.response,
		Done:     true,
		FullText: m.response,
		Err:      m.err,
	}
	close(ch)
	return ch, nil
}

func TestLLMEvaluatorPolicy(t *testing.T) {
	tests := []struct {
		name          string
		modelResponse string
		wantError     bool
	}{
		{
			name:          "Safe response",
			modelResponse: "SAFE",
			wantError:     false,
		},
		{
			name:          "Verbose safe response",
			modelResponse: "This text is SAFE.",
			wantError:     false,
		},
		{
			name:          "Unsafe response",
			modelResponse: "UNSAFE",
			wantError:     true,
		},
		{
			name:          "Verbose unsafe response",
			modelResponse: "This violates guidelines, so it is UNSAFE.",
			wantError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &mockSafetyModel{response: tt.modelResponse}
			policy := NewLLMEvaluatorPolicy(model, "")

			// We only care how it interprets the model's response to the original text
			err := policy.Validate(context.Background(), "Some text to evaluate")

			if (err != nil) != tt.wantError {
				t.Errorf("Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}
