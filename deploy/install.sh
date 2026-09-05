#!/usr/bin/env bash
# ResumeVoice AI — public/internet deployment provisioning.
# Run as root:  sudo bash deploy/install.sh
# Edits the 3 variables below, then run again if anything changes.
set -euo pipefail

APP_DIR="/home/shinde/Desktop/ResumeVoice_AI"
PUBLIC_IP="${RESUMEVOICE_PUBLIC_IP:-202.164.135.184}"
LAN_IP="${RESUMEVOICE_LAN_IP:-192.168.1.189}"
DOMAIN="${RESUMEVOICE_DOMAIN:-}"                 # e.g. voice.example.com
TURN_PASS="${RESUMEVOICE_TURN_PASSWORD:-CHANGE_ME_TURN_PASSWORD}"
TURN_USER="${RESUMEVOICE_TURN_USERNAME:-resumevoice}"

if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi

echo "==> 1/6 port check (8000 must be free for the systemd unit)"
if ss -ltn | grep -q ':8000 '; then
  echo "WARNING: something already listens on :8000 (a dev server?)."
  echo "  Stop it first (e.g. kill the uvicorn) or the unit below will fail to bind."
  echo "  Current owner: $(ss -ltnp | grep ':8000 ' | grep -o 'pid=[0-9]*' || echo none)"
fi

echo "==> 2/6 install packages (coturn, caddy)"
apt-get update -y
apt-get install -y coturn || true
if ! command -v caddy >/dev/null; then
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl 2>/dev/null || true
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg 2>/dev/null || true
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/deb.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null 2>&1 || true
  apt-get update -y
  apt-get install -y caddy || true
fi

echo "==> 3/6 firewall (Debian ufw)"
ufw allow 8000/tcp comment 'ResumeVoice app + wss signaling'
ufw allow 443/tcp comment 'Caddy TLS'
ufw allow 3478/tcp comment 'TURN'
ufw allow 3478/udp comment 'TURN'
ufw allow 49160:49200/udp comment 'TURN relay'
ufw --force enable

echo "==> 4/6 TURN (coturn)"
sed -e "s/user=resumevoice:CHANGE_ME_TURN_PASSWORD/user=${TURN_USER}:${TURN_PASS}/" \
    -e "s/external-ip=202.164.135.184\/192.168.1.189/external-ip=${PUBLIC_IP}\/${LAN_IP}/" \
    "$APP_DIR/deploy/turnserver.conf" > /etc/turnserver.conf
echo 'ENABLED=1' > /etc/default/coturn
systemctl enable coturn >/dev/null 2>&1 || true
systemctl restart coturn || true

echo "==> 5/6 Caddy (TLS reverse proxy)"
install -m 0644 "$APP_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile
mkdir -p /etc/caddy
if [[ -n "$DOMAIN" ]]; then
  echo "RESUMEVOICE_DOMAIN=${DOMAIN}" > /etc/caddy/caddy.env
else
  rm -f /etc/caddy/caddy.env
fi
systemctl enable caddy >/dev/null 2>&1 || true
systemctl restart caddy || true

echo "==> 6/6 systemd service"
install -m 0644 "$APP_DIR/deploy/resumevoice.service" /etc/systemd/system/resumevoice.service
systemctl daemon-reload
systemctl enable resumevoice >/dev/null 2>&1 || true
systemctl restart resumevoice || true
systemctl --no-pager status resumevoice --lines=5 || true

cat <<EOF

DONE. Endpoints:
  same-LAN : ws://${LAN_IP}:8000/ws/signal         (also https://${LAN_IP}:443/webrtc)
  public   : ws://${PUBLIC_IP}:8000/ws/signal      (needs router to fwd tcp 8000)
  public   : https://${DOMAIN:-<IP>}/webrtc        (TLS via Caddy)
  TURN     : turn:${PUBLIC_IP}:3478 user=${TURN_USER} pass=${TURN_PASS}

ROUTER (required for internet access from ANOTHER machine):
  1. Forward TCP 8000  -> ${LAN_IP}:8000   (signaling without TLS)
  2. Forward TCP/UDP 443 -> skip; Caddy handles TLS on :443 -> you may not need
     a router rule if the aim is wss: forward TCP/UDP 443 -> ${LAN_IP}:443
  3. Forward UDP 3478    -> ${LAN_IP}:3478     (TURN entry)
  4. Forward UDP 49160-49200 -> ${LAN_IP}      (TURN relay traffic)

APP SIDE (in ${APP_DIR}/.env):
  RESUMEVOICE_TURN_URL=turn:${PUBLIC_IP}:3478
  RESUMEVOICE_TURN_USERNAME=${TURN_USER}
  RESUMEVOICE_TURN_CREDENTIAL=${TURN_PASS}
EOF