#!/bin/bash
DIR="$(dirname "$0")"

set -a
source "$DIR/.env"
set +a

source "$DIR/venv/bin/activate"

# Open terminal 1: FastAPI server
gnome-terminal --title="AutoVideoEdit Server" -- bash -c "
  set -a; source '$DIR/.env'; set +a
  source '$DIR/venv/bin/activate'
  uvicorn server:app --host 0.0.0.0 --port 8001
  exec bash
"

# Kill any previous ngrok3 tunnel before starting
pkill -f "ngrok3.yml" 2>/dev/null; sleep 1

# Open terminal 2: ngrok
gnome-terminal --title="ngrok Tunnel" -- bash -c "
  ngrok start auto-video-edit --config ~/.config/ngrok/ngrok3.yml
  exec bash
"

echo "Both terminals launched."
