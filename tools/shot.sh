#!/bin/bash
# render SVG files to PNG with the bundled headless Chromium (window larger than the SVG, then crop)
CH=${CHROME:-chromium}
for f in "$@"; do
  f=$(readlink -f "$f")
  w=$(grep -o 'width="[0-9]*"' "$f" | head -1 | grep -o '[0-9]*'); h=$(grep -o 'height="[0-9]*"' "$f" | head -1 | grep -o '[0-9]*')
  timeout 60 "$CH" --headless --no-sandbox --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --window-size=$((w+40)),$((h+300)) --screenshot="${f%.svg}.png" "file://$f" >/dev/null 2>&1
  python3 -c "from PIL import Image; im=Image.open('${f%.svg}.png'); im.crop((0,0,$w*2,$h*2)).save('${f%.svg}.png')"
done
