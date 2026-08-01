"""Vyrobí široké (1×2) varianty dvoupanelových obrázků experimentu 7.

Původní obrázky z notebooku mají dva panely nad sebou. Do práce se vkládají
varianty s panely vedle sebe, protože se tak vejdou na stránku k tabulkám.
Skript pracuje čistě obrazově (žádný experiment se nespouští, čísla se
nemění): najde poslední zcela bílý vodorovný pás mezi panely, obrázek
rozřízne a půlky slepí vedle sebe se zarovnáním na spodní okraj.

Spouštění (po každém přegenerování původních PNG z notebooku):
    python3 background/make_exp7_q1_wide.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

FIG_DIR = Path(__file__).resolve().parent.parent / "figures" / "GP2_exp7_misspecification"
SOURCES = [
    "fig_exp7_q1_mse_vs_n.png",
    "fig_exp7_q2_mse_vs_ls.png",
]
GAP_PX = 40  # bílá mezera mezi panely ve výsledném obrázku


def find_split_row(img: Image.Image) -> int:
    """Najde řez v posledním zcela bílém pásu v prostřední části obrázku.

    Poslední pás se bere proto, aby popisek osy x horního panelu
    (leží mezi panely) zůstal u horního panelu.
    """
    a = np.asarray(img.convert("L"))
    h = a.shape[0]
    dark_per_row = (a < 220).sum(axis=1)
    runs = []
    start = None
    for i in range(int(h * 0.3), int(h * 0.7)):
        if dark_per_row[i] == 0:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, int(h * 0.7) - 1))
    runs = [r for r in runs if r[1] - r[0] > 3]
    if not runs:
        raise SystemExit("Mezi panely není čistě bílý pás — zkontroluj obrázek ručně.")
    lo, hi = runs[-1]
    return (lo + hi) // 2


def make_wide(src: Path) -> None:
    img = Image.open(src)
    dpi = img.info.get("dpi", (100, 100))
    split = find_split_row(img)
    top = img.crop((0, 0, img.width, split))
    bottom = img.crop((0, split, img.width, img.height))

    height = max(top.height, bottom.height)
    canvas = Image.new("RGB", (top.width + GAP_PX + bottom.width, height), "white")
    # zarovnání na spodní okraj, aby osy x obou panelů lícovaly
    canvas.paste(top, (0, height - top.height))
    canvas.paste(bottom, (top.width + GAP_PX, height - bottom.height))
    dst = src.with_name(src.stem + "_wide.png")
    canvas.save(dst, dpi=dpi)
    print(f"OK: {dst.name} ({canvas.width}x{canvas.height} px, řez na řádku {split})")


if __name__ == "__main__":
    for name in SOURCES:
        make_wide(FIG_DIR / name)
