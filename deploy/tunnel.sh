#!/usr/bin/env bash
# ResumeVoice AI — instant public HTTPS tunnel (cloudflared quick tunnel).
# Gives a public wss:// URL in ~1 minute with a valid cert, no router config.
# NOTE: quick tunnels proxy TCP only, so WebRTC *audio* media will not cross;
# use it to integrate text/events/SSE, then switch to Caddy+coturn (install.sh)
# for full mic->TTS audio over the internet.
# Run:  bash deploy/tunnel.sh
set -euo pipefail

APP_HOST="${RESUMEVOICE_HOST:-127.0.0.1}"
APP_PORT="${RESUMEVOICE_PORT:-8000}"
BIN="$HOME/.local/bin/cloudflared"

command -v "$BIN" >/dev/null 2>&1 || {
  mkdir -p "$HOME/.local/bin"
  echo "downloading cloudflared..."
  curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o "$BIN"
  chmod +x "$BIN"
}

echo "tunnel to http://${APP_HOST}:${APP_PORT} (Ctrl+C to stop)"
echo "find the public URL below, then open  <url>/webrtc  (wss:// signaling)"
"$BIN" tunnel --url "http://${APP_HOST}:${APP_PORT}"