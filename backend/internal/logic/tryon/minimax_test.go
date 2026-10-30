package tryon

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
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

// TestIsSensitiveInputError 验证能识别 MiniMax 1026 输入侧内容审核错误，
// 且不会把其他错误误判成审核错误（否则会走错降级路径）。
func TestIsSensitiveInputError(t *testing.T) {
	sensitive := []error{
		fmt.Errorf("MiniMax API 错误: input new_sensitive"),
		fmt.Errorf("MiniMax API 错误: INPUT NEW_SENSITIVE"),
		fmt.Errorf("status_code 1026"),
	}
	for _, err := range sensitive {
		if !isSensitiveInputError(err) {
			t.Errorf("expected sensitive-input error for %v", err)
		}
	}
	others := []error{
		nil,
		fmt.Errorf("无法连接模型服务"),
		fmt.Errorf("MiniMax API 错误: invalid params, image_reference must be one"),
		fmt.Errorf("MiniMax API 错误: insufficient balance"),
	}
	for _, err := range others {
		if isSensitiveInputError(err) {
			t.Errorf("must not treat as sensitive-input error: %v", err)
		}
	}
}

// TestCropTopRatio 验证裁剪只保留顶部指定比例，且对 WebP/JPEG/PNG 都能解码。
func TestCropTopRatio(t *testing.T) {
	// 构造一张 100x200 的红色 PNG。
	img := image.NewRGBA(image.Rect(0, 0, 100, 200))
	for y := 0; y < 200; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 255, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	src := base64.StdEncoding.EncodeToString(buf.Bytes())

	cropped, err := cropTopRatio(src, 0.5)
	if err != nil {
		t.Fatalf("cropTopRatio returned error: %v", err)
	}
	raw, err := base64.StdEncoding.DecodeString(cropped)
	if err != nil {
		t.Fatalf("cropped is not valid base64: %v", err)
	}
	ci, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("cropped is not a decodable image: %v", err)
	}
	b := ci.Bounds()
	if b.Dx() != 100 || b.Dy() != 100 {
		t.Errorf("expected 100x100 crop, got %dx%d", b.Dx(), b.Dy())
	}

	// 非法比例必须报错而不是 panic。
	if _, err := cropTopRatio(src, 0); err == nil {
		t.Error("expected error for ratio 0")
	}
	if _, err := cropTopRatio(src, 1.5); err == nil {
		t.Error("expected error for ratio > 1")
	}
	// 非法 base64 必须报错。
	if _, err := cropTopRatio("!!!not-base64!!!", 0.5); err == nil {
		t.Error("expected error for invalid base64")
	}
}

// TestMiniMaxFallsBackWhenSubjectRejected 验证参考图被内容审核拦截时的降级链路：
// 原图 → 裁剪头部重试 → 纯文生图。
// 必须最终出图，而不是把 1026 直接抛给用户。
func TestMiniMaxFallsBackWhenSubjectRejected(t *testing.T) {
	var mu sync.Mutex
	var reqs []minimaxImageRequest
	// 记录是否已经用过「完整人像」。第一次带完整人像 -> 拒绝；
	// 后续（裁剪后的或不带图的）-> 放行。
	rejected := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"The outfit consists of a red coat."}}]}`))
			return
		}
		var body minimaxImageRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		reqs = append(reqs, body)
		first := !rejected && len(body.SubjectRef) > 0
		if first {
			rejected = true
		}
		mu.Unlock()

		if first {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"base_resp":{"status_code":1026,"status_msg":"input new_sensitive"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"image_base64":["aW1hZ2U="]},"base_resp":{"status_code":0,"status_msg":"success"}}`))
	}))
	defer server.Close()

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "image-01",
	})

	// 构造一张 200x400 的 PNG 作为人像。
	img := image.NewRGBA(image.Rect(0, 0, 200, 400))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode failed: %v", err)
	}
	big := base64.StdEncoding.EncodeToString(buf.Bytes())

	result, err := gen.Generate(context.Background(), big, testPNGBase64, "orig prompt", "1024x1536")
	if err != nil {
		t.Fatalf("Generate must fall back instead of failing, got: %v", err)
	}
	if result != "aW1hZ2U=" {
		t.Fatalf("unexpected image result: %q", result)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(reqs) < 2 {
		t.Fatalf("expected at least 2 image calls (original + fallback), got %d", len(reqs))
	}
	// 第一次必须带参考图。
	if len(reqs[0].SubjectRef) != 1 {
		t.Errorf("first call must carry subject_reference")
	}
	// 后续调用不得再发送原图（那会重复被拒）。
	for i, rq := range reqs[1:] {
		if len(rq.SubjectRef) > 0 && len(rq.SubjectRef[0].ImageFile) == len(reqs[0].SubjectRef[0].ImageFile) &&
			rq.SubjectRef[0].ImageFile == reqs[0].SubjectRef[0].ImageFile {
			t.Errorf("call %d must not resend the rejected original image", i+2)
		}
	}
}

// TestMiniMaxDoesNotFallbackOnOtherErrors 验证非审核类错误不做降级，
// 直接返回错误（避免无意义的重试放大延迟）。
func TestMiniMaxDoesNotFallbackOnOtherErrors(t *testing.T) {
	var mu sync.Mutex
	calls := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"The outfit consists of a red coat."}}]}`))
			return
		}
		mu.Lock()
		calls++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"base_resp":{"status_code":2013,"status_msg":"invalid params"}}`))
	}))
	defer server.Close()

	gen := NewMiniMax(&ModelConfig{
		Name:    "minimax",
		APIKey:  "test-key",
		BaseURL: server.URL,
		Model:   "image-01",
	})

	if _, err := gen.Generate(context.Background(), testPNGBase64, testPNGBase64, "p", "1024x1536"); err == nil {
		t.Fatal("expected error for non-sensitivity failure")
	}
	mu.Lock()
	defer mu.Unlock()
	if calls != 1 {
		t.Fatalf("expected exactly 1 image call (no fallback), got %d", calls)
	}
}

// TestMiniMaxReportsMode 验证 mode 正确反映实际走的生成路径。
// 这是「结果是不是降级出来的」这个问题的唯一权威来源 ——
// 之前降级是静默发生的，用户看到成功结果却无从判断它是哪条路径产出的。
func TestMiniMaxReportsMode(t *testing.T) {
	visionOK := `{"choices":[{"message":{"content":"adult woman, oval face, dark hair"}}]}`
	okImage := `{"data":{"image_base64":["aW1hZ2U="]},"base_resp":{"status_code":0,"status_msg":"success"}}`

	// 构造一张可裁剪的 PNG（1x1 的图裁不出头部，裁剪会失败）。
	croppable := func() string {
		img := image.NewRGBA(image.Rect(0, 0, 200, 400))
		for y := 0; y < 400; y++ {
			for x := 0; x < 200; x++ {
				img.Set(x, y, color.RGBA{R: 200, G: 150, B: 120, A: 255})
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatalf("encode failed: %v", err)
		}
		return base64.StdEncoding.EncodeToString(buf.Bytes())
	}()

	// 场景 1：正常路径 —— 第一次图生图即通过
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req minimaxImageRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "MiniMax-Text-01" {
			_, _ = w.Write([]byte(visionOK))
			return
		}
		_, _ = w.Write([]byte(okImage))
	}))
	defer srv1.Close()

	m1 := NewMiniMax(&ModelConfig{Name: "minimax", APIKey: "k", BaseURL: srv1.URL, Model: "image-01"})
	if _, err := m1.Generate(context.Background(), croppable, testPNGBase64, "p", "1024x1536"); err != nil {
		t.Fatalf("场景1 生成失败: %v", err)
	}
	if got := m1.Mode(); got != "reference" {
		t.Errorf("场景1 mode 应为 reference，实际为 %q", got)
	}

	// 场景 2：裁剪路径 —— 第一���（完整人像）被 1026 拦，裁剪后的头部放行
	var mu2 sync.Mutex
	imageCalls2 := 0
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req minimaxImageRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "MiniMax-Text-01" {
			_, _ = w.Write([]byte(visionOK))
			return
		}
		mu2.Lock()
		imageCalls2++
		n := imageCalls2
		mu2.Unlock()
		if n == 1 {
			_, _ = w.Write([]byte(`{"base_resp":{"status_code":1026,"status_msg":"input new_sensitive"}}`))
			return
		}
		_, _ = w.Write([]byte(okImage))
	}))
	defer srv2.Close()

	m2 := NewMiniMax(&ModelConfig{Name: "minimax", APIKey: "k", BaseURL: srv2.URL, Model: "image-01"})
	if _, err := m2.Generate(context.Background(), croppable, testPNGBase64, "p", "1024x1536"); err != nil {
		t.Fatalf("场景2 生成失败: %v", err)
	}
	if got := m2.Mode(); got != "reference_cropped" {
		t.Errorf("场景2 mode 应为 reference_cropped，实际为 %q", got)
	}
	mu2.Lock()
	defer mu2.Unlock()
	if imageCalls2 < 2 {
		t.Errorf("场景2 应至少发起过 2 次图像请求（原图 + 裁剪图），实际 %d 次", imageCalls2)
	}

	// 场景 3：完全降级 —— 连裁剪后的头部也被拦，最终走纯文生图
	srv3 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req minimaxImageRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model == "MiniMax-Text-01" {
			_, _ = w.Write([]byte(visionOK))
			return
		}
		// 只有完全不带图的请求放行
		if len(req.SubjectRef) == 0 {
			_, _ = w.Write([]byte(okImage))
			return
		}
		_, _ = w.Write([]byte(`{"base_resp":{"status_code":1026,"status_msg":"input new_sensitive"}}`))
	}))
	defer srv3.Close()

	m3 := NewMiniMax(&ModelConfig{Name: "minimax", APIKey: "k", BaseURL: srv3.URL, Model: "image-01"})
	if _, err := m3.Generate(context.Background(), croppable, testPNGBase64, "p", "1024x1536"); err != nil {
		t.Fatalf("场景3 生成失败: %v", err)
	}
	if got := m3.Mode(); got != "text_only" {
		t.Errorf("场景3 mode 应为 text_only，实际为 %q", got)
	}
}
