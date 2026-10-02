#!/bin/sh
# Runs on the ml-brain server from cron every 5 minutes:
#   */5 * * * * $HOME/a3-car-price/auto_update.sh >> $HOME/a3-car-price/auto_update.log 2>&1
#
# GitHub Actions pushes a new image to Docker Hub only after the unit tests
# pass. This script pulls that image and recreates the container if the image
# changed (docker compose up -d does nothing when it is already up to date).
cd "$(dirname "$0")" || exit 1
docker compose pull -q
docker compose up -d
docker image prune -f > /dev/null
