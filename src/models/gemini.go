package models

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

type GeminiLLM struct {
	Client       *genai.Client
	Model        string
	PromptPrefix string
}

func NewGeminiLLM(ctx context.Context, model string, promptPrefix string) (Agent, error) {
	cl, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, err
	}
	return &GeminiLLM{Client: cl, Model: model, PromptPrefix: promptPrefix}, nil
}

func (g *GeminiLLM) Generate(ctx context.Context, prompt string) (any, error) {
	full := prompt
	if g.PromptPrefix != "" {
		full = g.PromptPrefix + "\n\n" + prompt
	}

	contents := []*genai.Content{genai.NewContentFromText(full, genai.RoleUser)}
	resp, err := g.Client.Models.GenerateContent(ctx, g.Model, contents, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini generate: %w", err)
	}

	if resp.Text() == "" {
		return nil, errors.New("gemini: empty response")
	}
	return resp.Text(), nil
}

// GenerateStream uses Gemini's streaming API to yield tokens incrementally.
func (g *GeminiLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	full := prompt
	if g.PromptPrefix != "" {
		full = g.PromptPrefix + "\n\n" + prompt
	}
	contents := []*genai.Content{genai.NewContentFromText(full, genai.RoleUser)}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		var sb strings.Builder
		for resp, err := range g.Client.Models.GenerateContentStream(ctx, g.Model, contents, nil) {
			if err != nil {
				// io.EOF signals normal end of stream
				if err.Error() == "no more items in iterator" || err == context.Canceled {
					ch <- StreamChunk{Done: true, FullText: sb.String()}
					return
				}
				ch <- StreamChunk{Done: true, FullText: sb.String(), Err: err}
				return
			}
			sb.WriteString(resp.Text())
			ch <- StreamChunk{Delta: resp.Text()}
		}
	}()

	return ch, nil
}

// NEW: pass images/videos as parts so Gemini can read them.
// Falls back to text-only if there are no binary attachments.
// gemini.go (inside package models)

func (g *GeminiLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	norm := make([]File, 0, len(files))
	for _, f := range files {
		normalizedMIME := normalizeMIME(f.Name, f.MIME)
		norm = append(norm, File{
			Name: f.Name,
			MIME: normalizedMIME,
			Data: f.Data,
		})
	}

	text := combinePromptWithFiles(prompt, norm)

	var parts []*genai.Part
	if p := strings.TrimSpace(g.PromptPrefix); p != "" {
		parts = append(parts, genai.NewPartFromText(p))
	}
	parts = append(parts, genai.NewPartFromText(text))

	for _, f := range norm {
		if len(f.Data) == 0 {
			continue
		}
		sanitized := sanitizeForGemini(f.MIME)
		if sanitized == "" {
			continue
		}
		parts = append(parts, genai.NewPartFromBytes(f.Data, sanitized))
	}

	contents := []*genai.Content{genai.NewContentFromParts(parts, genai.RoleUser)}
	resp, err := g.Client.Models.GenerateContent(ctx, g.Model, contents, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini generateWithFiles: %w", err)
	}
	return resp.Text(), nil
}
