package tryon

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// 1x1 的 PNG（透明像素），用作最小合法图片负载。
const testPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

// fakeMiniMaxServer 搭建一个模拟 MiniMax 的测试服务器。
// 视觉端点返回可控描述，生图端点记录收到的请求体。
type fakeMiniMaxServer struct {
	*httptest.Server

	mu          sync.Mutex
	imageReq    minimaxImageRequest
	imageCalls  int
	visionCalls int
	visionReply map[string]string // 依据提示词关键字返回的描述
	visionFail  bool
}

func newFakeMiniMax(t *testing.T, visionReply map[string]string) *fakeMiniMaxServer {
	t.Helper()
	f := &fakeMiniMaxServer{visionReply: visionReply}
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.visionCalls++
		fail := f.visionFail
		f.mu.Unlock()

		if fail {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		var body minimaxVisionRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		// 默认返回空串：未匹配到关键字时视为「该次视觉调用没有产出描述」，
		// 用于模拟人像描述失败、服装描述失败的场景。
		reply := ""
		if len(body.Messages) > 0 && len(body.Messages[0].Content) > 0 {
			text := body.Messages[0].Content[0].Text
			for key, val := range f.visionReply {
				if strings.Contains(text, key) {
					reply = val
					break
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{"content": reply},
			}},
		})
	})

	mux.HandleFunc("/v1/image_generation", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		f.imageCalls++
		var body minimaxImageRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		f.imageReq = body
		f.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":      map[string]any{"image_base64": []string{"aW1hZ2U="}},
			"base_resp": map[string]any{"status_code": 0, "status_msg": "success"},
		})
	})

	f.Server = httptest.NewServer(mux)
	t.Cleanup(f.Server.Close)
	return f
}

func (f *fakeMiniMaxServer) snapshot() (minimaxImageRequest, int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.imageReq, f.imageCalls, f.visionCalls
}

// TestMiniMaxSendsPersonAsSubjectReference 验证人像通过 subject_reference 传入，
// 且只传一张 —— MiniMax 传两张会返回 2013 "image_reference must be one"。
func TestMiniMaxSendsPersonAsSubjectReference(t *testing.T) {
	fake := newFakeMiniMax(t, map[string]string{
		"clothing items": "The outfit consists of a red coat, black skirt and white sneakers.",
		"the person":     "A young woman with an oval face and dark hair tied back.",
	})

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: fake.URL,
		Model:   "image-01",
	})

	result, err := gen.Generate(context.Background(), testPNGBase64, testPNGBase64, "orig prompt", "1024x1536")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if result != "aW1hZ2U=" {
		t.Fatalf("unexpected image result: %q", result)
	}

	req, imageCalls, visionCalls := fake.snapshot()
	if imageCalls != 1 {
		t.Fatalf("expected exactly 1 image generation call, got %d", imageCalls)
	}
	if visionCalls != 2 {
		t.Fatalf("expected 2 vision calls (identity + garment), got %d", visionCalls)
	}
	if len(req.SubjectRef) != 1 {
		t.Fatalf("subject_reference must contain exactly 1 image, got %d", len(req.SubjectRef))
	}
	if req.SubjectRef[0].Type != "character" {
		t.Fatalf("expected subject_reference type 'character', got %q", req.SubjectRef[0].Type)
	}
	if !strings.HasPrefix(req.SubjectRef[0].ImageFile, "data:image/") {
		t.Fatalf("subject_reference image_file must be a data URL, got %.40q", req.SubjectRef[0].ImageFile)
	}
}

// TestMiniMaxPromptHasSingleSubject 是本问题的核心回归测试。
//
// 旧实现把「照片里的人」和「展示服装的人」两个主语写进提示词，
// image-01 会把两个主语都画出来 —— 画面出现两个人，且原人像原封不动。
// 新实现必须保证提示词里只有一个人物主语，且服装描述里不出现人物词。
func TestMiniMaxPromptHasSingleSubject(t *testing.T) {
	fake := newFakeMiniMax(t, map[string]string{
		"clothing items": "The outfit consists of a red coat, black skirt and white sneakers.",
		"the person":     "A young woman with an oval face and dark hair tied back.",
	})

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: fake.URL,
		Model:   "image-01",
	})

	if _, err := gen.Generate(context.Background(), testPNGBase64, testPNGBase64, "orig prompt", "1024x1536"); err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	req, _, _ := fake.snapshot()

	// 旧实现特有的元指代：新提示词不得再出现 "Image 1" / "Image 2"。
	for _, forbidden := range []string{"Image 1", "Image 2", "the first image", "the second image"} {
		if strings.Contains(req.Prompt, forbidden) {
			t.Errorf("prompt must not contain meta reference %q:\n%s", forbidden, req.Prompt)
		}
	}

	// 提示词必须显式声明单人。
	if !strings.Contains(req.Prompt, "Exactly one person in the frame") {
		t.Errorf("prompt must assert a single person:\n%s", req.Prompt)
	}

	// 服装描述块不得出现人物主语（否则模型会画出第二个人）。
	// 只取 "OUTFIT TO WEAR ...:" 之后、下一行之前的服装描述本身，
	// 避免把模板里的 "Exactly one person in the frame" 误判进来。
	garmentPart := extractGarmentDescription(req.Prompt)
	if garmentPart == "" {
		t.Fatalf("prompt has no OUTFIT TO WEAR section:\n%s", req.Prompt)
	}
	for _, forbidden := range []string{"woman", "mannequin", "person", "girl", "model"} {
		if strings.Contains(strings.ToLower(garmentPart), forbidden) {
			t.Errorf("garment description must not mention %q:\n%s", forbidden, garmentPart)
		}
	}

	// 服装描述必须真的被用上。
	if !strings.Contains(req.Prompt, "red coat") {
		t.Errorf("garment description missing from prompt:\n%s", req.Prompt)
	}
	// 身份描述必须被用上。
	if !strings.Contains(req.Prompt, "oval face") {
		t.Errorf("identity description missing from prompt:\n%s", req.Prompt)
	}
}

// TestMiniMaxFailsWhenGarmentDescriptionUnavailable 验证服装描述拿不到时必须报错，
// 而不是退回纯文生图凭空捏一套衣服。
func TestMiniMaxFailsWhenGarmentDescriptionUnavailable(t *testing.T) {
	fake := newFakeMiniMax(t, nil)
	fake.visionFail = true

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: fake.URL,
		Model:   "image-01",
	})

	_, err := gen.Generate(context.Background(), testPNGBase64, testPNGBase64, "orig prompt", "1024x1536")
	if err == nil {
		t.Fatal("expected error when garment description is unavailable")
	}
	if !strings.Contains(err.Error(), "服装图解析失败") {
		t.Fatalf("unexpected error message: %v", err)
	}
	if _, imageCalls, _ := fake.snapshot(); imageCalls != 0 {
		t.Fatalf("must not call image generation without a garment description, got %d calls", imageCalls)
	}
}

// TestMiniMaxIdentityFailureStillProceeds 验证只有身份描述失败时仍能出图
// （身份描述是可选增强，服装描述才是必需项）。
func TestMiniMaxIdentityFailureStillProceeds(t *testing.T) {
	fake := newFakeMiniMax(t, map[string]string{
		"clothing items": "The outfit consists of a red coat and white sneakers.",
	})

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: fake.URL,
		Model:   "image-01",
	})

	// 人像用无效负载，让该次视觉调用返回空描述。
	result, err := gen.Generate(context.Background(), "not-base64!!", testPNGBase64, "orig prompt", "1024x1536")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if result != "aW1hZ2U=" {
		t.Fatalf("unexpected image result: %q", result)
	}
	req, imageCalls, _ := fake.snapshot()
	if imageCalls != 1 {
		t.Fatalf("expected 1 image call, got %d", imageCalls)
	}
	if strings.Contains(req.Prompt, "MODEL IDENTITY") {
		t.Errorf("prompt should omit identity block when identity description is empty:\n%s", req.Prompt)
	}
	if !strings.Contains(req.Prompt, "red coat") {
		t.Errorf("garment description missing:\n%s", req.Prompt)
	}
}

// TestMiniMaxUsesRealImageMIME 验证 Data URL 的 MIME 标签取自真实字节，
// 而不是一律硬编码 image/png（前端上传的可能是 JPEG / PNG / WebP）。
func TestMiniMaxUsesRealImageMIME(t *testing.T) {
	// 最小合法 JPEG 头（SOI + APP0/JFIF）
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00}
	got := detectImageMIME(encodeB64(jpeg))
	if got != "image/jpeg" {
		t.Fatalf("expected image/jpeg, got %q", got)
	}

	if got := detectImageMIME(testPNGBase64); got != "image/png" {
		t.Fatalf("expected image/png, got %q", got)
	}

	if got := detectImageMIME("!!!not-base64!!!"); got != "image/png" {
		t.Fatalf("expected fallback image/png, got %q", got)
	}
}

// TestResolveUsedPrompt 验证接口返回的是实际使用的提示词，而不是构建阶段的模板。
func TestResolveUsedPrompt(t *testing.T) {
	gen := NewMiniMax(&ModelConfig{Name: "minimax", APIKey: "k", BaseURL: "http://x", Model: "image-01"})
	gen.usedPrompt = "actual prompt"
	if got := resolveUsedPrompt(gen, "template prompt"); got != "actual prompt" {
		t.Fatalf("expected actual prompt, got %q", got)
	}

	// 未实现 PromptReporter 的适配器应退回模板提示词。
	if got := resolveUsedPrompt(&stubGenerator{}, "template prompt"); got != "template prompt" {
		t.Fatalf("expected fallback template prompt, got %q", got)
	}
}

type stubGenerator struct{}

func (s *stubGenerator) Generate(ctx context.Context, personB64, garmentB64, prompt, size string) (string, error) {
	return "", nil
}
func (s *stubGenerator) Name() string { return "stub" }

// extractGarmentDescription 从最终提示词里取出服装描述本身。
func extractGarmentDescription(prompt string) string {
	const marker = "OUTFIT TO WEAR (she is dressed in exactly these items): "
	idx := strings.Index(prompt, marker)
	if idx < 0 {
		return ""
	}
	rest := prompt[idx+len(marker):]
	if nl := strings.Index(rest, "\n"); nl >= 0 {
		rest = rest[:nl]
	}
	return strings.TrimSpace(rest)
}
