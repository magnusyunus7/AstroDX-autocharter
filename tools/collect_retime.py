"""Copy the songs whose charts' beats don't follow the music (found by
tools/check_codec.py) into one folder, so they can be retimed by hand.

    python -m tools.collect_retime "G:\\My Drive\\maisquared"

For each song: maidata.txt and track.mp3 (plus bg if there is one), into
data/retime/<group>/<version> - <song>/, and a README.txt listing which
difficulties are off and how well they fit (share of notes within 3 ms of a
1/48-beat or 5th/7th/9th/10th position on the chart's own beat grid).

  1_clearly_off   some difficulty below 90%: tempo changes squeezed into odd
                  divisors by the old converter, or notes off the beat nearly
                  everywhere (offset or a deliberate gimmick)
  2_slightly_off  90-98%: probably a few odd spots only, worth a look
Songs already copied are skipped, so it can be stopped and run again.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (version/song, [(difficulty slot, fit)])
CLEARLY_OFF = [
    ('01. maimai/Chu Chu', [(5, 0.445)]),
    ('01. maimai/Sun Dance', [(6, 0.563)]),
    ('01. maimai/ジングルベル', [(5, 0.8)]),
    ('01. maimai/炭★坑★節', [(5, 0.71)]),
    ('02. maimai PLUS/Space Harrier Main Theme [Reborn]', [(5, 0.693)]),
    ('02. maimai PLUS/御旗のもとに', [(4, 0.89)]),
    ('03. GreeN/I ♥', [(5, 0.202)]),
    ('03. GreeN/残酷な天使のテーゼ', [(1, 0.0), (2, 0.041), (3, 0.059), (4, 0.119), (5, 0.304)]),
    ('04. GreeN PLUS/からくりピエロ', [(6, 0.455)]),
    ('05. ORANGE/おても☆Yan', [(5, 0.483)]),
    ('06. ORANGE PLUS/ってゐ！ ～えいえんてゐVer～', [(1, 0.821), (2, 0.84), (3, 0.858), (4, 0.842), (5, 0.886), (6, 0.847)]),
    ('07. PiNK/Wonderland Wars オープニング', [(1, 0.882), (2, 0.882), (3, 0.833), (4, 0.885), (5, 0.917), (6, 0.581)]),
    ('09. MURASAKi/DETARAME ROCK&ROLL THEORY', [(1, 0.923), (2, 0.927), (3, 0.876), (4, 0.899), (5, 0.882)]),
    ('09. MURASAKi/だんだん早くなる', [(1, 0.882), (2, 0.911), (3, 0.898), (4, 0.887), (5, 0.868), (6, 0.893)]),
    ('19. UNiVERSE PLUS/Transcend Lights', [(3, 0.597), (4, 0.316), (5, 0.418)]),
    ('20. FESTiVAL/Blank Paper (Prod. TEMPLIME)', [(5, 0.421)]),
    ('20. FESTiVAL/Catch Me If You Can', [(5, 0.874)]),
    ('22. BUDDiES/ツギハギスタッカート', [(4, 0.788)]),
]
SLIGHTLY_OFF = [
    ('01. maimai/ZIGG-ZAGG', [(1, 0.969)]),
    ('02. maimai PLUS/DO RORO DERODERO ON DO RORO', [(5, 0.961)]),
    ('03. GreeN/円舞曲、君に', [(5, 0.975)]),
    ('03. GreeN/神室雪月花', [(3, 0.978), (4, 0.977), (5, 0.976)]),
    ('04. GreeN PLUS/sweet little sister', [(1, 0.97), (3, 0.978), (4, 0.964), (5, 0.969)]),
    ('07. PiNK/かくしん的☆めたまるふぉ～ぜっ！', [(3, 0.976), (4, 0.975), (5, 0.977)]),
    ('07. PiNK/東方妖々夢 ～the maximum moving about～', [(1, 0.957), (2, 0.953), (3, 0.968), (4, 0.963)]),
    ('08. PiNK PLUS/Mr. Wonderland', [(5, 0.907)]),
    ('09. MURASAKi/夜明けまであと３秒', [(5, 0.968)]),
    ('10. MURASAKi PLUS/ホシトハナ', [(2, 0.937), (3, 0.922), (4, 0.925), (5, 0.925)]),
    ('17. Splash PLUS/もぺもぺ', [(5, 0.976)]),
    ('17. Splash PLUS/星めぐり、果ての君へ。', [(6, 0.946)]),
    ('19. UNiVERSE PLUS/チューリングの跡', [(5, 0.965)]),
    ('25. PRiSM PLUS/Amereistr', [(4, 0.966), (5, 0.912)]),
    ('28. MAGiCAL/PPPP (feat. Hatsune Miku, Kasane Teto)', [(5, 0.971)]),
]
SLOT_NAMES = {1: "Easy", 2: "Basic", 3: "Advanced", 4: "Expert", 5: "Master", 6: "Re:Master"}


def copy(src: Path, dst: Path) -> None:
    """Plain read/write copy: Google Drive's streamed files refuse Windows'
    fast copy (shutil.copy2 -> WinError 1). Written under a temporary name
    first, so a stopped copy never looks finished."""
    tmp = dst.with_name(dst.name + ".part")
    with open(src, "rb") as a, open(tmp, "wb") as b:
        shutil.copyfileobj(a, b, 64 * 1024)
    tmp.replace(dst)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).expanduser()
    if not src.is_dir():
        sys.exit(f"not a folder: {src}")
    out = ROOT / "data" / "retime"
    readme = ["Charts whose beats don't follow the music (fit = share of notes on clean beat positions).", ""]
    t0, n, failed = time.time(), 0, []
    for group, songs in (("1_clearly_off", CLEARLY_OFF), ("2_slightly_off", SLIGHTLY_OFF)):
        readme.append(f"== {group}")
        for song, slots in songs:
            version, name = song.split("/", 1)
            dst = out / group / f"{version} - {name}"
            dst.mkdir(parents=True, exist_ok=True)
            here = src / version / name
            for f in ["maidata.txt", "track.mp3"] + [p.name for p in here.glob("bg.*")]:
                if not (here / f).exists():
                    print(f"  ! missing {song}/{f}")
                    continue
                if (dst / f).exists() and (dst / f).stat().st_size == (here / f).stat().st_size:
                    continue                                         # copied on an earlier run
                for attempt in (1, 2, 3):
                    try:
                        copy(here / f, dst / f)
                        break
                    except OSError as e:
                        print(f"  ! {song}/{f}: {e} (try {attempt}/3)", flush=True)
                        time.sleep(5 * attempt)
                else:
                    failed.append(f"{song}/{f}")
            n += 1
            diffs = ", ".join(f"{SLOT_NAMES.get(s, s)} {fit:.0%}" for s, fit in slots)
            readme.append(f"  {version} - {name}: {diffs}")
            print(f"  {n}/{len(CLEARLY_OFF) + len(SLIGHTLY_OFF)} {song} ({time.time() - t0:.0f} s)", flush=True)
        readme.append("")
    (out / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print(f"done: {out}")
    if failed:
        print(f"{len(failed)} file(s) couldn't be read from Drive (run again later, or download them by hand):")
        for f in failed:
            print("   ", f)


if __name__ == "__main__":
    main()
