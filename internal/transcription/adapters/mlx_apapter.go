package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"scriberr/internal/transcription/interfaces"
	"scriberr/internal/transcription/registry"
)

// MLXAdapter implements the TranscriptionAdapter interface for Apple MLX
type MLXAdapter struct {
	*BaseAdapter
	envPath string
}

// NewMLXAdapter creates a new MLX adapter
func NewMLXAdapter(envPath string) *MLXAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:     "mlx_whisper",
		ModelFamily: "whisper",
		DisplayName: "Apple MLX Whisper",
		Description: "Optimized Whisper models for Apple Silicon",
		Version:     "1.0.0",
		SupportedLanguages: []string{"auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "zh", "ko"},
		SupportedFormats:   []string{"wav", "mp3", "flac", "m4a"},
		RequiresGPU:        false, // MLX uses Unified Memory / Neural Engine
		MemoryRequirement:  4096,  // MLX models can be memory hungry depending on quantization
		Features: map[string]bool{
			"timestamps": true,
			"word_level": true,
			"fast_mode":  true,
		},
		Metadata: map[string]string{
			"engine":   "mlx",
			"platform": "darwin", // Only works on macOS
		},
	}

	schema := []interfaces.ParameterSchema{
		{
			Name:        "model",
			Type:        "string",
			Required:    false,
			Default:     "mlx-community/whisper-large-v3-mlx",
			Options:     []string{"mlx-community/whisper-large-v3-mlx", "mlx-community/whisper-large-v3-turbo", "mlx-community/whisper-base-mlx"},
			Description: "Hugging Face model ID for MLX",
			Group:       "basic",
		},
		{
			Name:        "quantization",
			Type:        "string",
			Required:    false,
			Default:     "4bit",
			Options:     []string{"4bit", "8bit", "none"},
			Description: "Model quantization level",
			Group:       "advanced",
		},
	}

	// Adjust base path as needed
	baseAdapter := NewBaseAdapter("mlx_whisper", filepath.Join(envPath, "MLX"), capabilities, schema)

	return &MLXAdapter{
		BaseAdapter: baseAdapter,
		envPath:     envPath,
	}
}

func (m *MLXAdapter) GetSupportedModels() []string {
	return []string{"mlx-community/whisper-large-v3-mlx", "mlx-community/whisper-base-mlx"}
}

func (m *MLXAdapter) PrepareEnvironment(ctx context.Context) error {
	// Only proceed if running on macOS
	if runtime.GOOS != "darwin" {
		return fmt.Errorf("MLX adapter is only supported on macOS")
	}

	mlxPath := filepath.Join(m.envPath, "MLX")

	// Check if already ready
	if CheckEnvironmentReady(mlxPath, "import mlx_whisper") {
		m.initialized = true
		return nil
	}

	// Create directory
	if err := os.MkdirAll(mlxPath, 0755); err != nil {
		return err
	}

	// Check if pyproject.toml exists to avoid re-initializing
	if _, err := os.Stat(filepath.Join(mlxPath, "pyproject.toml")); os.IsNotExist(err) {
		// Initialize UV project with a specific name to avoid shadowing 'mlx' package
		initCmd := exec.Command("uv", "init", "--name", "scriberr-mlx-wrapper")
		initCmd.Dir = mlxPath
		if out, err := initCmd.CombinedOutput(); err != nil {
			return fmt.Errorf("uv init failed: %s: %w", string(out), err)
		}
	}

	// Install dependencies
	installCmd := exec.Command("uv", "add", "mlx-whisper", "ffmpeg-python")
	installCmd.Dir = mlxPath
	if out, err := installCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to install mlx-whisper: %s", string(out))
	}

	m.initialized = true
	return nil
}

func (m *MLXAdapter) Transcribe(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	startTime := time.Now()
	m.LogProcessingStart(input, procCtx)
	defer func() { m.LogProcessingEnd(procCtx, time.Since(startTime), nil) }()

	if err := m.ValidateAudioInput(input); err != nil {
		return nil, err
	}

	tempDir, err := m.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, err
	}
	defer m.CleanupTempDirectory(tempDir)

	// Create the Python script
	scriptPath := filepath.Join(tempDir, "transcribe_mlx.py")
	scriptContent := m.generatePythonScript()
	if err := os.WriteFile(scriptPath, []byte(scriptContent), 0644); err != nil {
		return nil, fmt.Errorf("failed to write script: %w", err)
	}

	modelName := m.GetStringParameter(params, "model")
	outputJson := filepath.Join(tempDir, "output.json")

	// Construct UV command
	mlxPath := filepath.Join(m.envPath, "MLX")
	cmd := exec.CommandContext(ctx, "uv", "run", "--project", mlxPath, "python", scriptPath,
		"--audio", input.FilePath,
		"--model", modelName,
		"--output", outputJson,
	)

	// Set standard output for logging
	logFile, _ := os.Create(filepath.Join(procCtx.OutputDirectory, "mlx_transcription.log"))
	cmd.Stdout = logFile
	cmd.Stderr = logFile

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("MLX execution failed: %w", err)
	}

	return m.parseResult(outputJson, params)
}

func (m *MLXAdapter) parseResult(jsonPath string, params map[string]interface{}) (*interfaces.TranscriptResult, error) {
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, err
	}

	var mlxOutput struct {
		Text     string `json:"text"`
		Segments []struct {
			Start float64 `json:"start"`
			End   float64 `json:"end"`
			Text  string  `json:"text"`
		} `json:"segments"`
		Language string `json:"language"`
	}

	if err := json.Unmarshal(data, &mlxOutput); err != nil {
		return nil, fmt.Errorf("failed to parse MLX output: %w", err)
	}

	// Convert to standard interface
	result := &interfaces.TranscriptResult{
		Text:      mlxOutput.Text,
		Language:  mlxOutput.Language,
		ModelUsed: m.GetStringParameter(params, "model"),
		Segments:  make([]interfaces.TranscriptSegment, len(mlxOutput.Segments)),
	}

	for i, seg := range mlxOutput.Segments {
		result.Segments[i] = interfaces.TranscriptSegment{
			Start: seg.Start,
			End:   seg.End,
			Text:  strings.TrimSpace(seg.Text),
		}
	}

	return result, nil
}

// Helper: Python script to bridge MLX and our JSON format
// We include clean_obj to handle NaN/Infinity values which crash Go's JSON parser
func (m *MLXAdapter) generatePythonScript() string {
	return `
import argparse
import json
import mlx_whisper
import math

def clean_obj(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_obj(v) for v in obj]
    return obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    
    # Transcribe
    result = mlx_whisper.transcribe(
        args.audio, 
        path_or_hf_repo=args.model,
        word_timestamps=True
    )

    # Clean NaNs/Infs which cause JSON errors in Go/other parsers
    result = clean_obj(result)

    # Save to JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
`
}

// Auto-register
func init() {
	registry.RegisterTranscriptionAdapter("mlx_whisper", NewMLXAdapter("./data/mlx-env"))
}
