// 同源代理：浏览器 → https://<站点>/api/tryon（Cloudflare 边缘）
//          → Cloudflare Tunnel → 源站 → Go 后端 (127.0.0.1:8000)
//
// 为什么需要这一层（两层原因，都实测过）：
//   1. HTTPS 页面不能直连 http:// 后端。浏览器以 Mixed Content 硬性拦截，
//      请求根本发不出去，页面表现为「无法连接换装服务」。
//   2. Cloudflare Workers/Pages Functions 的 fetch 禁止 IP 字面量
//      （实测一律 error code 1003），所以边缘层也不能直连源站 IP。
//
// 剩下的唯一通路是 Cloudflare Tunnel：边缘主动连出到源站，不经过公网 DNS 解析，
// 也不受 Workers 的 IP 禁令约束（走 Cloudflare 内部路由）。
//
// 鉴权不在这里做：Authorization 头原样透传，由源站的 Bearer Token 中间件校验。
const TUNNEL_HOST = 'api.nvzhuang.fenghuashao.top';

export async function onRequestPost({ request }) {
  const headers = { 'Content-Type': 'application/json' };
  const auth = request.headers.get('Authorization');
  if (auth) headers['Authorization'] = auth;

  try {
    const upstream = await fetch(`https://${TUNNEL_HOST}/tryon`, {
      method: 'POST',
      headers,
      body: await request.text(),
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    });
  } catch (e) {
    return new Response(
      JSON.stringify({
        code: 502,
        message: '后端代理不可达，请确认源站 Tunnel 域名与 DNS 记录已配置',
        data: null,
      }),
      { status: 502, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

export async function onRequestOptions() {
  return new Response(null, { status: 204 });
}
