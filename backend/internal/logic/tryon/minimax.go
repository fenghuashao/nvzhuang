package tryon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// MiniMax 图像生成适配器（image-01 系列）。
//
// 协议特点（2026-10 实测）：
//   - 文生图：POST {baseURL}/v1/image_generation
//     body: {model, prompt, aspect_ratio, response_format}
//     resp: {"data":{"image_base64":[...]}, "base_resp":{"status_code":0}}
//   - 图生图：同样端点，但只能传 **一张** 参考图（subject_reference 数组 length 必须为 1，
//     传两张会返回 status_code 2013 "image_reference must be one"）。
//
// 换装任务天然需要「人像 + 服装」两张参考图，因此这里复用 gpt_image.go 已验证的思路：
//  1. 先调用多模态 chat/completions，让视觉模型读取两张图并输出结构化视觉描述；
//  2. 再用该描述做文生图（image-01 不带参考图），得到换装结果。
//
// 这条链路已实测：纯色测试图 → 正确识别出「粉色上衣 + 藏青下装」并生成对应全身照。
type MiniMax struct {
	apiKey  string
	baseURL string
	model   string
	// visionModel 用于第一步的视觉描述；为空时回退到 MiniMax-Text-01。
	visionModel string
	name        string
	client      *http.Client
}

func NewMiniMax(cfg *ModelConfig) *MiniMax {
	return &MiniMax{
		apiKey:      cfg.APIKey,
		baseURL:     strings.TrimRight(cfg.BaseURL, "/"),
		model:       cfg.Model,
		visionModel: cfg.VisionModel,
		name:        cfg.Name,
		client:      newModelHTTPClient(cfg.ClientProvided),
	}
}

func (m *MiniMax) Name() string {
	if m.name == "" {
		return "minimax"
	}
	return m.name
}

// minimaxImageRequest 文生图请求体。
type minimaxImageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	AspectRatio    string `json:"aspect_ratio,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
	Seed           int64  `json:"seed,omitempty"`
	N              int    `json:"n,omitempty"`
}

type minimaxImageResponse struct {
	Data *struct {
		ImageBase64 []string `json:"image_base64"`
		URLs        []string `json:"urls"`
	} `json:"data"`
	BaseResp *struct {
		StatusCode int    `json:"status_code"`
		StatusMsg  string `json:"status_msg"`
	} `json:"base_resp"`
}

// minimaxVisionRequest 多模态对话请求体（用于提取两张参考图的视觉描述）。
type minimaxVisionRequest struct {
	Model     string                 `json:"model"`
	Messages  []minimaxVisionMessage `json:"messages"`
	MaxTokens int                    `json:"max_tokens,omitempty"`
}

type minimaxVisionMessage struct {
	Role    string                 `json:"role"`
	Content []minimaxVisionContent `json:"content"`
}

type minimaxVisionContent struct {
	Type     string           `json:"type"`
	Text     string           `json:"text,omitempty"`
	ImageURL *minimaxImageURL `json:"image_url,omitempty"`
}

type minimaxImageURL struct {
	URL string `json:"url"`
}

type minimaxVisionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// visionDescribePrompt 让视觉模型分别描述人像与服装，供文生图还原。
const visionDescribePrompt = "Describe Image 1 (the person: adult East Asian female, approximate age, hair style and color, facial features, expression) and Image 2 (the exact clothing outfit: every single garment item, colors, patterns, materials, cut, hem length, shoes, accessories). Return a concise 2-sentence visual description for a fashion catalog image generator."

// tryOnBuildPrompt 把视觉描述拼成最终文生图提示词。
const tryOnBuildPrompt = `Full-body commercial fashion catalog photograph:
%s
Strict requirements:
1. The adult female model is wearing the complete outfit described above.
2. Fully clothed, modest neckline, standard tailored fit, upright standing posture.
3. Clean daylight studio, neutral minimal background, photorealistic, crisp fabric textures.
Output: ONLY the final dressed-up photograph.`

// Generate 执行换装：视觉描述 → 文生图。
func (m *MiniMax) Generate(ctx context.Context, personB64, garmentB64, prompt, size string) (string, error) {
	if m.apiKey == "" {
		return "", fmt.Errorf("MiniMax API Key 未配置（llm.models.%s.apiKey 或环境变量 MINIMAX_API_KEY）", m.name)
	}
	if m.model == "" {
		return "", fmt.Errorf("MiniMax 未配置 model 名（llm.models.%s.model）", m.name)
	}

	// 步骤 1：视觉描述。失败不致命 —— 退回用原始提示词做文生图。
	desc := m.describeVisualFeatures(ctx, personB64, garmentB64)
	finalPrompt := prompt
	if desc != "" {
		finalPrompt = fmt.Sprintf(tryOnBuildPrompt, desc)
	}

	// 步骤 2：文生图。
	reqBody := minimaxImageRequest{
		Model:          m.model,
		Prompt:         finalPrompt,
		AspectRatio:    normalizeMiniMaxAspect(size),
		ResponseFormat: "base64",
		N:              1,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.baseURL+"/v1/image_generation", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("模型服务请求创建失败")
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("无法连接模型服务")
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("MiniMax API 错误 (status %d): %s", resp.StatusCode, truncate(string(respBody), 500))
	}

	var result minimaxImageResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", err
	}
	// base_resp.status_code 非 0 时 HTTP 仍是 200，必须单独判断。
	if result.BaseResp != nil && result.BaseResp.StatusCode != 0 {
		return "", fmt.Errorf("MiniMax API 错误: %s", result.BaseResp.StatusMsg)
	}
	if result.Data == nil || len(result.Data.ImageBase64) == 0 {
		return "", fmt.Errorf("MiniMax API 未返回图片")
	}
	return result.Data.ImageBase64[0], nil
}

// describeVisualFeatures 调用多模态模型读取两张参考图并返回视觉描述。
// 任何错误都返回空串（调用方会退回原始提示词）。
func (m *MiniMax) describeVisualFeatures(ctx context.Context, personB64, garmentB64 string) string {
	visionModel := m.visionModel
	if visionModel == "" {
		visionModel = "MiniMax-Text-01"
	}

	reqBody := minimaxVisionRequest{
		Model: visionModel,
		Messages: []minimaxVisionMessage{
			{
				Role: "user",
				Content: []minimaxVisionContent{
					{Type: "text", Text: visionDescribePrompt},
					{Type: "image_url", ImageURL: &minimaxImageURL{URL: "data:image/png;base64," + ensureBase64(personB64)}},
					{Type: "image_url", ImageURL: &minimaxImageURL{URL: "data:image/png;base64," + ensureBase64(garmentB64)}},
				},
			},
		},
		MaxTokens: 300,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return ""
	}

	// 视觉分析最长 15 秒，避免长时间等待。
	vCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(vCtx, "POST", m.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ""
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	resp, err := m.client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	var res minimaxVisionResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return ""
	}
	if len(res.Choices) > 0 {
		return strings.TrimSpace(res.Choices[0].Message.Content)
	}
	return ""
}

// normalizeMiniMaxAspect 把 OpenAI 风格尺寸转成 MiniMax 的 aspect_ratio。
// MiniMax 支持 1:1 / 16:9 / 4:3 / 3:2 / 2:3 / 3:4 / 9:16 / 21:9。
func normalizeMiniMaxAspect(size string) string {
	switch size {
	case "1024x1536", "1792x1024", "auto":
		// 换装默认竖版 2:3（832x1248）
		return "2:3"
	case "1536x1024":
		return "3:2"
	case "1024x1024":
		return "1:1"
	default:
		return "2:3"
	}
}
