#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${LAN_HOST:?Set LAN_HOST to the LAN IPv4 address of this computer}"
export LAN_HOST
export LAN_PORT="${LAN_PORT:-8443}"
umask 077
mkdir -p .lan
chmod 700 .lan
if [[ ! -f .lan/invite-password ]]; then
  openssl rand -hex 24 > .lan/invite-password
fi
export LAN_INVITE_PASSWORD
LAN_INVITE_PASSWORD="$(cat .lan/invite-password)"
# Create a short-lived certificate for this exact LAN address on each launch.
openssl req -x509 -newkey rsa:3072 -nodes -keyout .lan/key.pem -out .lan/cert.pem \
  -days 7 -subj '/CN=BioPrint local study' -addext "subjectAltName=IP:${LAN_HOST}" 2>/dev/null
openssl x509 -in .lan/cert.pem -noout -fingerprint -sha256
printf 'Participant URL: https://%s:%s\nInvitation username: participant\nInvitation password: %s/.lan/invite-password\n' "$LAN_HOST" "$LAN_PORT" "$PWD"
exec conda run --no-capture-output -n bigidea python -m uvicorn lan_gateway:create_app --factory \
  --host "$LAN_HOST" --port "$LAN_PORT" --no-proxy-headers --no-access-log \
  --ssl-keyfile .lan/key.pem --ssl-certfile .lan/cert.pem \
  --limit-concurrency 32 --timeout-keep-alive 5
