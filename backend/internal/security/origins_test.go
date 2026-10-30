package security

import (
	"strings"
	"testing"
)

// TestDefaultOriginsExcludeUpstreamSite 锁定 CORS 默认白名单不包含上游作者的站点。
//
// 上游项目（paipiange.github.io）曾被写进默认白名单。本项目已自建部署，
// 保留该条目等于允许上游站点从浏览器调用本机后端 —— 属于不必要的信任面。
// 本测试防止它被再次加回默认值。
//
// 注意：LoadSecurityConfig 在配置缺少 security 段时回退到的正是
// DefaultSecurityConfig()，所以本测试同时覆盖了「默认值」与「回退值」两条路径。
func TestDefaultOriginsExcludeUpstreamSite(t *testing.T) {
	cfg := DefaultSecurityConfig()
	for _, o := range cfg.TrustedOrigins {
		if strings.Contains(o, "paipiange.github.io") {
			t.Fatalf("默认 CORS 白名单不应包含上游站点 %q，实际为 %v", o, cfg.TrustedOrigins)
		}
	}
}

// TestDefaultOriginsAllowLocalhost 确认本地调试来源仍然放行。
func TestDefaultOriginsAllowLocalhost(t *testing.T) {
	cfg := DefaultSecurityConfig()
	want := map[string]bool{
		"http://localhost:3000": false,
		"http://127.0.0.1:3000": false,
	}
	for _, o := range cfg.TrustedOrigins {
		if _, ok := want[o]; ok {
			want[o] = true
		}
	}
	for o, found := range want {
		if !found {
			t.Errorf("默认白名单应包含 %q，实际为 %v", o, cfg.TrustedOrigins)
		}
	}
}
