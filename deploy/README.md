# ResumeVoice AI — deployment to other machines

Three tiers, pick what fits your test:

## 1. Same LAN/VPN — works right now, nothing to install

The server already listens on `0.0.0.0:8000`. From any machine on the LAN:

```
ws://192.168.1.189:8000/ws/signal
```

> Browser mic (getUserMedia) only works from a `localhost` or HTTPS origin.
> For LAN browser mic use Caddy (tier 2/3 below) or the Chrome flag
> `unsafely-treat-insecure-origin-as-secure` + `http://192.168.1.189:8000/webrtc`.

## 2. Public internet — full setup (TLS + TURN, browser-mic ready)

Two manual steps only YOU can do, then run the installer:

1. **Router** — log into your router and forward:
   | Port(s) | Type | → LAN IP |
   |---|---|---|
   | 8000 | TCP | 192.168.1.189 |
   | 443 | TCP/UDP | 192.168.1.189 |
   | 3478 | TCP/UDP | 192.168.1.189 |
   | 49160–49200 | UDP | 192.168.1.189 |
2. **Run the installer** (top-of-file vars first):
   ```
   sudo RESUMEVOICE_DOMAIN=voice.example.com \
        RESUMEVOICE_TURN_PASSWORD=YourStrongPass \
        bash deploy/install.sh
   ```
   No domain? Leave `RESUMEVOICE_DOMAIN` empty — Caddy uses a self-signed cert
   on `:443`; signaling works, but browser mic will still be blocked (any
   browser) because the cert is untrusted. A real domain is the only clean path
   to mic in browsers.
3. **Point DNS** at the public IP if you set a domain.

Then set in the repo `.env` (service picks it up on restart):
```
RESUMEVOICE_TURN_URL=turn:202.164.135.184:3478
RESUMEVOICE_TURN_USERNAME=resumevoice
RESUMEVOICE_TURN_CREDENTIAL=<YourStrongPass>
RESUMEVOICE_CORS_ORIGINS=https://voice.example.com
```

Endpoint that the platform team uses:
```
wss://voice.example.com/ws/signal   (or ws://202.164.135.184:8000/ws/signal)
```
Test page: `https://voice.example.com/webrtc` (or `https://202.164.135.184/webrtc`).

## 3. Instant tunnel — no router, no domain (integration only)

```
bash deploy/tunnel.sh
```
Cloudflare gives a public `https://<random>.trycloudflare.com` URL with a real
cert in ~1 minute. `wss://<random>.trycloudflare.com/ws/signal` works for
**text + events + SSE** from anywhere. Media (mic → TTS audio) will NOT cross a
quick tunnel (TCP-only); that tier needs n.2.

## Interaction with the currently running dev server

- The `install.sh` systemd unit (`resumevoice.service`) binds the same
  `:8000`. If a manually-started uvicorn is running, stop it first or the unit
  fails to bind. The unit runs the *same* interpreter
  (`/usr/local/bin/python3.12`, edit the `RESUMEVOICE_PYTHON` var if yours
  differs).

## Docker (docker compose) details

- The compose `resumevoice` service talks to Ollama at
  `http://host.docker.internal:11434/v1`. Ollama must listen on **0.0.0.0**,
  not just loopback, or the container answers `{"type":"error","text":"Connection error."}`:
  ```
  sudo systemctl edit ollama      # add:
  # [Service]
  # Environment="OLLAMA_HOST=0.0.0.0"
  sudo systemctl restart ollama
  ```
  No sudo available? Run the included bridge instead (same effect, no root):
  ```
  python3 deploy/llm_bridge.py --listen 172.17.0.1 --listen-port 11434 &
  ```
  It forwards the Docker bridge gateway IP → `127.0.0.1:11434` so the
  container's `host.docker.internal` URL just works. Note it is a manual
  process (restart it after reboot, or apply the systemd fix above).
- The image needs `websockets>=12` (uvicorn's WebSocket protocol). Without it
  every `/ws/*` endpoint returns 404. It is in `pyproject.toml [all]` now; a
  fresh `docker compose build resumevoice` bakes it in. If you only `docker cp`
  files into a running container instead, run
  `docker exec resumevoice-app pip install "websockets>=12"` once.

## Files

```
deploy/
├── install.sh          # sudo: firewall + coturn + caddy + systemd
├── resumevoice.service # systemd unit for the app
├── Caddyfile           # TLS reverse proxy (wss:// + /webrtc page)
├── turnserver.conf     # coturn TURN relay
├── llm_bridge.py       # non-root Ollama-to-docker bridge (TCP forward)
└── tunnel.sh           # cloudflared quick tunnel (integration tier)
```