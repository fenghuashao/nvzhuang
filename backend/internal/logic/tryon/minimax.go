package tryon

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// MiniMax 图像生成适配器（image-01 系列）。
//
// 协议特点（2026-10 实测）：
//   - 文生图/图生图：POST {baseURL}/v1/image_generation
//     body: {model, prompt, aspect_ratio, response_format, subject_reference?}
//     resp: {"data":{"image_base64":[...]}, "base_resp":{"status_code":0}}
//   - subject_reference **只能传一张** 参考图：实测传两张（无论 type 如何组合）
//     一律返回 status_code 2013 "image_reference must be one"。
//   - 错误不在 HTTP 状态码上（恒 200），而在 body 的 base_resp.status_code。
//
// 换装实现（2026-10-30 重写）：
//
//	旧实现走「视觉描述 → 纯文生图」。问题是视觉描述模板同时描述了「照片里的人」
//	和「展示服装的人」两个主语，image-01 作为纯文生图模型会把两个主语都画出来 ——
//	表现为画面里出现两个人：左边原人像原封不动（白T恤+牛仔裤），右边才是换好装的。
//	这不是随机抽风，而是提示词结构决定的确定性结果（基线 2/2 复现两人）。
//	加 "replace" / "exactly one person" 这类祈使句无效 —— 模型数的是描述里的主语，不是指令。
//
//	现实现：
//	1. 人像通过 subject_reference 传入（模型直接拿到人脸与体态，细节保真度最好）；
//	2. 服装只做「纯服装描述」，明令不提及人物/人台，避免引入第二个主语；
//	3. 两者拼成单人文生图提示词，参考图同时用于锚定身份。
//
//	实测：单人 3/3、跨春夏秋冬 4 品类 4/4，参考图的眼镜等细节可保留。
type MiniMax struct {
	apiKey  string
	baseURL string
	model   string
	// visionModel 用于读取参考图并生成视觉描述；为空时回退到 MiniMax-Text-01。
	visionModel string
	name        string
	client      *http.Client
	// usedPrompt 记录本次实际使用的提示词。
	// 每个请求都会新建一个适配器实例（见 NewGenerator），因此不存在并发竞争。
	usedPrompt string
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

// UsedPrompt 返回本次实际发给模型的提示词。
// 接口返回的 prompt 字段此前一直是拼装前的模板，与真实提示词不符，排查时极具误导性。
func (m *MiniMax) UsedPrompt() string {
	return m.usedPrompt
}

// minimaxImageRequest 图生图/文生图请求体。
type minimaxImageRequest struct {
	Model          string              `json:"model"`
	Prompt         string              `json:"prompt"`
	AspectRatio    string              `json:"aspect_ratio,omitempty"`
	ResponseFormat string              `json:"response_format,omitempty"`
	Seed           int64               `json:"seed,omitempty"`
	N              int                 `json:"n,omitempty"`
	SubjectRef     []minimaxSubjectRef `json:"subject_reference,omitempty"`
}

// minimaxSubjectRef 单张参考图。MiniMax 只接受一张。
type minimaxSubjectRef struct {
	Type      string `json:"type"`
	ImageFile string `json:"image_file"`
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

// visionIdentityPrompt 提取人像的身份特征。
// 关键约束：不得提及任何服装，否则描述里会同时存在「原来穿什么」和「要穿什么」，
// 模型会把两者都画出来（表现为原服装残留）。
const visionIdentityPrompt = `Describe ONLY the person in this photo: apparent age, face shape, skin tone, eye shape, eyebrow shape, nose, mouth, hair style and hair color.
Do NOT describe or mention any clothing, outfit, garment or accessory the person is wearing.
Output one concise sentence about identity only, with no clothing words.`

// visionGarmentPrompt 提取服装细节。
// 关键约束：只描述服装本身，不得提及人物/人台/衣架。服装图往往带无头人台，
// 一旦描述里出现第二个人物主语，文生图就会画出两个人。
const visionGarmentPrompt = `Describe ONLY the clothing items in this image. List every garment item with its exact color, pattern, material, cut, neckline, sleeve length, hem length, layering order, plus shoes, socks, bag, hat and accessories.
IMPORTANT: describe ONLY the garments themselves. Do not mention any person, woman, model, mannequin, hanger, body, face, hair or hands, even if a mannequin or body is visible in the photo.
Write one sentence that starts with 'The outfit consists of' and contains no person words at all.`

// tryOnBuildPrompt 把身份描述与服装描述拼成最终提示词。
// 句式刻意保持「单人叙述」：全文只出现一个主语（the model）。
const tryOnBuildPrompt = `Fashion catalog photograph, single adult female model, full body, front view.
%s
OUTFIT TO WEAR (she is dressed in exactly these items): %s
The model wears the entire outfit listed above; none of her original clothing remains visible.
Clean daylight studio, plain light grey seamless background, photorealistic, crisp fabric texture.
Exactly one person in the frame.
Output: ONLY the final dressed-up photograph.`

// Generate 执行换装：人像走 subject_reference，服装走纯服装描述。
// 注意：prompt 入参在本适配器中不参与拼装 —— 最终提示词完全由
// 「身份描述 + 服装描述」构成，与上游模板无关（上游模板是给文生图模型的，
// 其中「第一张图/第二张图」的表述在这里没有对应输入）。
func (m *MiniMax) Generate(ctx context.Context, personB64, garmentB64, prompt, size string) (string, error) {
	if m.apiKey == "" {
		return "", fmt.Errorf("MiniMax API Key 未配置（llm.models.%s.apiKey 或环境变量 MINIMAX_API_KEY）", m.name)
	}
	if m.model == "" {
		return "", fmt.Errorf("MiniMax 未配置 model 名（llm.models.%s.model）", m.name)
	}

	// 步骤 1：视觉描述。人像身份描述为可选增强，服装描述为必需。
	identity, garment := m.describeVisualFeatures(ctx, personB64, garmentB64)
	if garment == "" {
		// 没有服装描述就只剩纯文生图，等于凭空捏一套衣服。这里直接失败，
		// 宁可让用户重试，也不返回一张与所选服装无关的图。
		return "", fmt.Errorf("服装图解析失败（视觉模型未返回有效结果）")
	}

	identityBlock := ""
	if identity != "" {
		identityBlock = "MODEL IDENTITY: " + identity + "\n"
	}
	finalPrompt := fmt.Sprintf(tryOnBuildPrompt, identityBlock, garment)
	m.usedPrompt = finalPrompt

	// 步骤 2：图生图。人像作为唯一参考图锚定身份与体态。
	reqBody := minimaxImageRequest{
		Model:          m.model,
		Prompt:         finalPrompt,
		AspectRatio:    normalizeMiniMaxAspect(size),
		ResponseFormat: "base64",
		N:              1,
		SubjectRef: []minimaxSubjectRef{
			{Type: "character", ImageFile: buildImageDataURL(personB64)},
		},
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

// describeVisualFeatures 并行调用多模态模型，分别提取人像身份描述与服装描述。
// 两个调用互不依赖，并行执行可把总耗时从约 5.5s 压到约 3s。
// 任一失败都返回空串：服装描述为空时调用方会直接报错，人像描述为空只影响身份精细度。
func (m *MiniMax) describeVisualFeatures(ctx context.Context, personB64, garmentB64 string) (identity, garment string) {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		identity = m.chatVision(ctx, visionIdentityPrompt, personB64)
	}()
	go func() {
		defer wg.Done()
		garment = m.chatVision(ctx, visionGarmentPrompt, garmentB64)
	}()
	wg.Wait()
	return identity, garment
}

// chatVision 调用多模态模型读取单张参考图并返回文本描述。
// 任何错误都返回空串（调用方按可选增强处理）。
func (m *MiniMax) chatVision(ctx context.Context, prompt, imageB64 string) string {
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
					{Type: "text", Text: prompt},
					{Type: "image_url", ImageURL: &minimaxImageURL{URL: buildImageDataURL(imageB64)}},
				},
			},
		},
		MaxTokens: 400,
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

// buildImageDataURL 把 base64 图片数据包装成 Data URL。
// MIME 由真实字节嗅探得出：前端上传的可能是 JPEG / PNG / WebP，
// 此前一律硬编码 image/png，属于与实际内容不符的标签。
func buildImageDataURL(b64 string) string {
	raw := ensureBase64(b64)
	return "data:" + detectImageMIME(raw) + ";base64," + raw
}

// detectImageMIME 嗅探 base64 图片数据的真实 MIME 类型。
// 只解码前 512 字节（http.DetectContentType 的最大探测长度），
// 避免为拿 MIME 而整图解码。嗅探失败时回退 image/png。
func detectImageMIME(b64 string) string {
	const sniffLen = 512
	// 每 4 个 base64 字符编码 3 字节，取足以覆盖 sniffLen 的最短 4 字符对齐前缀。
	prefixLen := ((sniffLen + 2) / 3) * 4
	if len(b64) > prefixLen {
		b64 = b64[:prefixLen]
	}
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "image/png"
	}
	mime := http.DetectContentType(raw)
	if strings.HasPrefix(mime, "image/") {
		return mime
	}
	return "image/png"
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
