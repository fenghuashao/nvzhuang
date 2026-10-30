package middleware

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/gogf/gf/v2/frame/g"
	"github.com/gogf/gf/v2/net/ghttp"

	"backend/internal/security"
)

func TestCORSOriginValidation(t *testing.T) {
	cfg := &security.SecurityConfig{
		TrustedOrigins: []string{
			"http://localhost:3000",
			"https://nvzhuang.fenghuashao.top",
		},
		AllowPrivateNetwork: true,
	}
	cors := NewCORSMiddleware(cfg)

	// 1. 白名单 Origin
	s := g.Server("test-cors-allowed")
	s.Group("/", func(group *ghttp.RouterGroup) {
		group.Middleware(RequestID, cors.Handler)
		group.ALL("/test", func(r *ghttp.Request) {
			r.Response.Write("ok")
		})
	})
	s.SetPort(0)
	_ = s.Start()
	defer s.Shutdown()

	client := &http.Client{}

	// 测试普通 GET 白名单
	req, _ := http.NewRequest(http.MethodGet, "http://127.0.0.1:"+strings.Split(s.GetListenedAddress(), ":")[1]+"/test", nil)
	req.Header.Set("Origin", "http://localhost:3000")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "http://localhost:3000" {
		t.Fatalf("unexpected allow origin: %s", resp.Header.Get("Access-Control-Allow-Origin"))
	}

	// 测试 OPTIONS 预检
	reqOptions, _ := http.NewRequest(http.MethodOptions, "http://127.0.0.1:"+strings.Split(s.GetListenedAddress(), ":")[1]+"/test", nil)
	reqOptions.Header.Set("Origin", "http://localhost:3000")
	reqOptions.Header.Set("Access-Control-Request-Private-Network", "true")
	respOptions, err := client.Do(reqOptions)
	if err != nil {
		t.Fatal(err)
	}
	defer respOptions.Body.Close()
	if respOptions.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for OPTIONS, got %d", respOptions.StatusCode)
	}
	if respOptions.Header.Get("Access-Control-Allow-Private-Network") != "true" {
		t.Fatalf("expected PNA allow header")
	}

	// 2. 非法 Origin
	reqBlocked, _ := http.NewRequest(http.MethodGet, "http://127.0.0.1:"+strings.Split(s.GetListenedAddress(), ":")[1]+"/test", nil)
	reqBlocked.Header.Set("Origin", "http://evil.com")
	respBlocked, err := client.Do(reqBlocked)
	if err != nil {
		t.Fatal(err)
	}
	defer respBlocked.Body.Close()
	if respBlocked.StatusCode != http.StatusForbidden {
		t.Fatalf("expected 403, got %d", respBlocked.StatusCode)
	}

	bodyBytes, _ := io.ReadAll(respBlocked.Body)
	t.Logf("respBlocked body: %s", string(bodyBytes))

	var errBody struct {
		Code int `json:"code"`
		Data struct {
			ErrorCode string `json:"error_code"`
			RequestID string `json:"request_id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(bodyBytes, &errBody)
	if errBody.Code != 40301 || errBody.Data.ErrorCode != "ORIGIN_NOT_ALLOWED" {
		t.Fatalf("unexpected error response: %+v", errBody)
	}
	if errBody.Data.RequestID == "" {
		t.Fatalf("expected request_id in error response")
	}
}

func TestAuthMiddleware(t *testing.T) {
	validToken := "valid-token-at-least-32-characters-long-123456"
	envKey := "TEST_MIDDLEWARE_KEYS"
	os.Setenv(envKey, validToken)
	defer os.Unsetenv(envKey)

	store, err := security.NewTokenStore(true, envKey)
	if err != nil {
		t.Fatal(err)
	}
	authMw := NewAuthMiddleware(store)

	s := g.Server("test-auth-mw")
	s.Group("/", func(group *ghttp.RouterGroup) {
		group.Middleware(RequestID, authMw.Handler)
		group.POST("/tryon", func(r *ghttp.Request) {
			r.Response.Write("success")
		})
	})
	s.SetPort(0)
	_ = s.Start()
	defer s.Shutdown()

	client := &http.Client{}
	port := strings.Split(s.GetListenedAddress(), ":")[1]

	// 1. 缺少 Token
	reqNoAuth, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", nil)
	respNoAuth, err := client.Do(reqNoAuth)
	if err != nil {
		t.Fatal(err)
	}
	defer respNoAuth.Body.Close()
	if respNoAuth.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", respNoAuth.StatusCode)
	}
	var res1 struct {
		Code int `json:"code"`
		Data struct {
			ErrorCode string `json:"error_code"`
		} `json:"data"`
	}
	_ = json.NewDecoder(respNoAuth.Body).Decode(&res1)
	if res1.Code != 40101 || res1.Data.ErrorCode != "AUTH_REQUIRED" {
		t.Fatalf("unexpected res1: %+v", res1)
	}

	// 2. 错误 Token
	reqBadAuth, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", nil)
	reqBadAuth.Header.Set("Authorization", "Bearer invalid-token-with-at-least-32-chars-long-000")
	respBadAuth, err := client.Do(reqBadAuth)
	if err != nil {
		t.Fatal(err)
	}
	defer respBadAuth.Body.Close()
	if respBadAuth.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", respBadAuth.StatusCode)
	}
	var res2 struct {
		Code int `json:"code"`
		Data struct {
			ErrorCode string `json:"error_code"`
		} `json:"data"`
	}
	_ = json.NewDecoder(respBadAuth.Body).Decode(&res2)
	if res2.Code != 40102 || res2.Data.ErrorCode != "AUTH_INVALID" {
		t.Fatalf("unexpected res2: %+v", res2)
	}

	// 3. 正确 Token
	reqValid, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", nil)
	reqValid.Header.Set("Authorization", "Bearer "+validToken)
	respValid, err := client.Do(reqValid)
	if err != nil {
		t.Fatal(err)
	}
	defer respValid.Body.Close()
	if respValid.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", respValid.StatusCode)
	}
}

func TestBodyLimitMiddleware(t *testing.T) {
	limitMw := NewBodyLimitMiddleware(100) // 100 bytes limit

	s := g.Server("test-body-limit")
	s.Group("/", func(group *ghttp.RouterGroup) {
		group.Middleware(RequestID, limitMw.Handler)
		group.POST("/tryon", func(r *ghttp.Request) {
			r.Response.Write("body received")
		})
	})
	s.SetPort(0)
	_ = s.Start()
	defer s.Shutdown()

	client := &http.Client{}
	port := strings.Split(s.GetListenedAddress(), ":")[1]

	// 超限请求体 (Content-Length > 100)
	largePayload := strings.Repeat("A", 200)
	req, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", strings.NewReader(largePayload))
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d", resp.StatusCode)
	}

	var res struct {
		Code int `json:"code"`
		Data struct {
			ErrorCode string `json:"error_code"`
		} `json:"data"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&res)
	if res.Code != 41301 || res.Data.ErrorCode != "REQUEST_TOO_LARGE" {
		t.Fatalf("unexpected 413 response: %+v", res)
	}
}

func TestRateLimitMiddleware(t *testing.T) {
	limiter := security.NewRateLimiter(60, 1, 10) // 1 burst
	rlMw := NewRateLimitMiddleware(limiter)

	s := g.Server("test-rate-limit-mw")
	s.Group("/", func(group *ghttp.RouterGroup) {
		group.Middleware(RequestID, rlMw.Handler)
		group.POST("/tryon", func(r *ghttp.Request) {
			r.Response.Write("ok")
		})
	})
	s.SetPort(0)
	_ = s.Start()
	defer s.Shutdown()

	client := &http.Client{}
	port := strings.Split(s.GetListenedAddress(), ":")[1]

	// 第一次：成功
	req1, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", nil)
	resp1, err := client.Do(req1)
	if err != nil {
		t.Fatal(err)
	}
	defer resp1.Body.Close()
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp1.StatusCode)
	}

	// 第二次：被限流 (429)
	req2, _ := http.NewRequest(http.MethodPost, "http://127.0.0.1:"+port+"/tryon", nil)
	resp2, err := client.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", resp2.StatusCode)
	}
	if resp2.Header.Get("Retry-After") == "" {
		t.Fatal("expected Retry-After header on 429")
	}

	var res struct {
		Code int `json:"code"`
		Data struct {
			ErrorCode string `json:"error_code"`
		} `json:"data"`
	}
	_ = json.NewDecoder(resp2.Body).Decode(&res)
	if res.Code != 42901 || res.Data.ErrorCode != "RATE_LIMITED" {
		t.Fatalf("unexpected 429 response: %+v", res)
	}
}
