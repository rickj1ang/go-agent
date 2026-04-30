//go:build integration

package models

import (
	"context"
	"strings"
	"testing"
)

func TestGeminiLive_Generate(t *testing.T) {
	ctx := context.Background()
	agent, err := NewGeminiLLM(ctx, "gemini-2.5-flash", "")
	if err != nil {
		t.Fatalf("NewGeminiLLM: %v", err)
	}

	resp, err := agent.Generate(ctx, "What is 2+2? Reply with just the number.")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	text, ok := resp.(string)
	if !ok {
		t.Fatalf("expected string, got %T", resp)
	}
	text = strings.TrimSpace(text)
	if text == "" {
		t.Fatal("empty response")
	}
	t.Logf("Generate response: %s", text)
}

func TestGeminiLive_GenerateWithFiles_Image(t *testing.T) {
	ctx := context.Background()
	agent, err := NewGeminiLLM(ctx, "gemini-2.5-flash", "")
	if err != nil {
		t.Fatalf("NewGeminiLLM: %v", err)
	}

	// 1x1 red pixel PNG
	png := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
		0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
		0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
		0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc,
		0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
		0x44, 0xae, 0x42, 0x60, 0x82,
	}

	files := []File{
		{Name: "pixel.png", MIME: "image/png", Data: png},
	}

	resp, err := agent.GenerateWithFiles(ctx, "Describe this image in one sentence.", files)
	if err != nil {
		t.Fatalf("GenerateWithFiles: %v", err)
	}

	text, ok := resp.(string)
	if !ok {
		t.Fatalf("expected string, got %T", resp)
	}
	text = strings.TrimSpace(text)
	if text == "" {
		t.Fatal("empty response")
	}
	t.Logf("GenerateWithFiles response: %s", text)
}

func TestGeminiLive_GenerateWithFiles_TextFile(t *testing.T) {
	ctx := context.Background()
	agent, err := NewGeminiLLM(ctx, "gemini-2.5-flash", "")
	if err != nil {
		t.Fatalf("NewGeminiLLM: %v", err)
	}

	files := []File{
		{Name: "data.txt", MIME: "text/plain", Data: []byte("The capital of France is Paris.")},
	}

	resp, err := agent.GenerateWithFiles(ctx, "What is the capital mentioned in the attached file? Reply with just the city name.", files)
	if err != nil {
		t.Fatalf("GenerateWithFiles: %v", err)
	}

	text, ok := resp.(string)
	if !ok {
		t.Fatalf("expected string, got %T", resp)
	}
	text = strings.TrimSpace(text)
	if text == "" {
		t.Fatal("empty response")
	}
	t.Logf("GenerateWithFiles (text) response: %s", text)
}

func TestGeminiLive_GenerateStream(t *testing.T) {
	ctx := context.Background()
	agent, err := NewGeminiLLM(ctx, "gemini-2.5-flash", "")
	if err != nil {
		t.Fatalf("NewGeminiLLM: %v", err)
	}

	ch, err := agent.GenerateStream(ctx, "Count from 1 to 5, one number per line.")
	if err != nil {
		t.Fatalf("GenerateStream: %v", err)
	}

	var full strings.Builder
	for chunk := range ch {
		if chunk.Err != nil {
			t.Fatalf("stream error: %v", chunk.Err)
		}
		full.WriteString(chunk.Delta)
		if chunk.Done && chunk.FullText == "" {
			t.Fatal("stream ended without FullText")
		}
	}

	result := full.String()
	if result == "" {
		t.Fatal("empty streamed response")
	}
	t.Logf("GenerateStream response: %s", result)
}
