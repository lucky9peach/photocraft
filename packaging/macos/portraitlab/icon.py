#!/usr/bin/env python3
"""Draw the PortraitLab app icon (rose→amber squircle with a portrait silhouette) and build a .icns."""
import os, subprocess, sys
from PIL import Image, ImageDraw, ImageFilter

OUT = os.environ["ICON_OUT"]
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

# Rounded-square plate (macOS style: inset, generous radius), vertical gradient.
inset, radius = 92, 232
plate = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(plate)
top, bottom = (244, 143, 168), (247, 205, 158)
grad = Image.new("RGBA", (1, S))
for y in range(S):
    t = y / (S - 1)
    grad.putpixel((0, y), (round(top[0] + (bottom[0] - top[0]) * t),
                           round(top[1] + (bottom[1] - top[1]) * t),
                           round(top[2] + (bottom[2] - top[2]) * t), 255))
grad = grad.resize((S, S))
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([inset, inset, S - inset, S - inset], radius=radius, fill=255)
plate.paste(grad, (0, 0), mask)
# Soft top highlight.
hl = Image.new("L", (S, S), 0)
ImageDraw.Draw(hl).ellipse([inset + 60, inset - 180, S - inset - 60, inset + 320], fill=70)
hl = hl.filter(ImageFilter.GaussianBlur(60))
plate.paste(Image.new("RGBA", (S, S), (255, 255, 255, 255)), (0, 0), Image.composite(hl, Image.new("L", (S, S), 0), mask))
img.alpha_composite(plate)

# Portrait silhouette: head + shoulders, in warm white.
fg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
f = ImageDraw.Draw(fg)
cx = S // 2
f.ellipse([cx - 132, 268, cx + 132, 532], fill=(255, 250, 245, 255))          # head
f.rounded_rectangle([cx - 232, 560, cx + 232, 812], radius=118, fill=(255, 250, 245, 255))  # shoulders
img.alpha_composite(fg)

img.save(os.path.join(OUT, "PortraitLab-1024.png"))

# .iconset → .icns
iconset = os.path.join(OUT, "PortraitLab.iconset")
os.makedirs(iconset, exist_ok=True)
for size in [16, 32, 64, 128, 256, 512]:
    img.resize((size, size), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}.png"))
    img.resize((size * 2, size * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(OUT, "PortraitLab.icns")], check=True)
print("wrote", os.path.join(OUT, "PortraitLab.icns"), os.path.getsize(os.path.join(OUT, "PortraitLab.icns")), "bytes")
