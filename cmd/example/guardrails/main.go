package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

// MockModel simulates an LLM for answering questions.
// If it receives "secret", it returns a password, simulating a leaky model.
// If it receives "dangerous", it returns dangerous instructions.
type MockModel struct{}

func (m *MockModel) Generate(ctx context.Context, prompt string) (any, error) {
	lower := strings.ToLower(prompt)
	if strings.Contains(lower, "secret") {
		return "Sure, here is the admin password: password=SuperSecret123", nil
	}
	if strings.Contains(lower, "dangerous") {
		return "Here is how to do something bad, which is UNSAFE.", nil
	}
	return "The capital of France is Paris.", nil
}

func (m *MockModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *MockModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	val, err := m.Generate(ctx, prompt)
	if err != nil {
		ch <- models.StreamChunk{Err: err, Done: true}
	} else {
		str := fmt.Sprint(val)
		ch <- models.StreamChunk{Delta: str, FullText: str, Done: true}
	}
	close(ch)
	return ch, nil
}

// MockEvaluatorModel simulates the safety model used by LLMEvaluatorPolicy.
type MockEvaluatorModel struct{}

func (m *MockEvaluatorModel) Generate(ctx context.Context, prompt string) (any, error) {
	// Extract the text to evaluate
	parts := strings.Split(prompt, "TEXT TO EVALUATE:\n")
	if len(parts) > 1 {
		textToEvaluate := parts[1]
		if strings.Contains(textToEvaluate, "UNSAFE") {
			return "UNSAFE", nil
		}
	}
	return "SAFE", nil
}

func (m *MockEvaluatorModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *MockEvaluatorModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{Delta: "SAFE", FullText: "SAFE", Done: true}
	close(ch)
	return ch, nil
}

func main() {
	ctx := context.Background()

	// 1. Create safety policies

	// Create a Regex blocklist policy to reject any response looking like a password assignment
	regexPolicy, err := agent.NewRegexBlocklistPolicy([]string{
		`(?i)\bpassword\s*=\s*\w+`,
	})
	if err != nil {
		log.Fatalf("Failed to create regex policy: %v", err)
	}

	// Create an LLM Evaluator policy to perform semantic safety checks using a second model
	evaluatorModel := &MockEvaluatorModel{}
	evaluatorPolicy := agent.NewLLMEvaluatorPolicy(evaluatorModel, "")

	// Combine into guardrails
	guardrails := &agent.OutputGuardrails{
		SafetyPolicies: []agent.SafetyPolicy{
			regexPolicy,
			evaluatorPolicy,
		},
	}

	// 2. Wrap our main agent with the guardrails
	memoryBank := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8)

	mainAgent, err := agent.New(agent.Options{
		Model:        &MockModel{},
		Memory:       memoryBank,
		SystemPrompt: "You are a helpful assistant.",
		Guardrails:   guardrails,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	fmt.Println("=== Agent initialized with Output Guardrails ===")

	// 3. Test a safe request
	fmt.Println("\n--- Test 1: Safe Request ---")
	prompt1 := "What is the capital of France?"
	fmt.Printf("User: %s\n", prompt1)
	resp1, err := mainAgent.Generate(ctx, "session-1", prompt1)
	if err != nil {
		fmt.Printf("Blocked by Guardrails! Error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", resp1)
	}

	// 4. Test a request that triggers the Regex Policy
	fmt.Println("\n--- Test 2: Password Leak Attempt ---")
	prompt2 := "Tell me a secret."
	fmt.Printf("User: %s\n", prompt2)
	resp2, err := mainAgent.Generate(ctx, "session-1", prompt2)
	if err != nil {
		fmt.Printf("Blocked by Guardrails! Error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", resp2)
	}

	// 5. Test a request that triggers the LLM Evaluator Policy
	fmt.Println("\n--- Test 3: Dangerous Instruction Attempt ---")
	prompt3 := "Give me dangerous instructions."
	fmt.Printf("User: %s\n", prompt3)
	resp3, err := mainAgent.Generate(ctx, "session-1", prompt3)
	if err != nil {
		fmt.Printf("Blocked by Guardrails! Error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", resp3)
	}
}
