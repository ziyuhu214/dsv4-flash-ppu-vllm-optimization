#!/bin/bash
# Resume download of T-HEAD/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken from ModelScope
set -u
REPO="T-HEAD/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken"
BASE="https://www.modelscope.cn/models/$REPO/resolve/master"
DEST="/mnt/cpfs/models/DeepSeek-V4-Flash-0731-Quant-W-INT8-PerChannel-A-INT8-PerToken"
MANIFEST="/workspace/pytorch/dsv4_int8_manifest.txt"   # lines: <path> <size>

download_one() {
  local path="$1" size="$2"
  local out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  # already complete?
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
    echo "OK(skip) $path"
    return 0
  fi
  local tmp="$out.incomplete"
  for attempt in 1 2 3 4 5; do
    curl -sSL -C - --retry 3 --retry-delay 5 --max-time 7200 -o "$tmp" "$BASE/$path"
    local got
    got=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
    if [ "$got" = "$size" ]; then
      mv "$tmp" "$out"
      echo "DONE $path"
      return 0
    fi
    echo "RETRY($attempt) $path got=$got want=$size"
    sleep 5
  done
  echo "FAIL $path"
  return 1
}

export -f download_one
export BASE DEST

# parallel over manifest, 4 at a time
FAILED=0
while read -r path size; do
  echo "$path $size"
done < "$MANIFEST" | xargs -P4 -n2 bash -c 'download_one "$0" "$1"' || FAILED=1

echo "=== verify all ==="
BAD=0
while read -r path size; do
  if [ ! -f "$DEST/$path" ] || [ "$(stat -c%s "$DEST/$path")" != "$size" ]; then
    echo "MISSING/BAD: $path"
    BAD=1
  fi
done < "$MANIFEST"
[ "$BAD" = 0 ] && echo "ALL FILES VERIFIED" || echo "SOME FILES FAILED"
