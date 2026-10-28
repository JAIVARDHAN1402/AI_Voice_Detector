#!/bin/bash
# resumable download that drops and re-opens stalled connections: fetch.sh URL OUTFILE
until curl -L --retry 10 --retry-delay 2 --retry-all-errors --speed-time 15 --speed-limit 20000 -C - -s -o "$2" "$1"; do sleep 2; done
echo "FETCHED $2"
