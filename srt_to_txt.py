from pathlib import Path
import re

def convert_folder_srt_to_txt(in_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    timer_pattern = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}")

    for srt_file in in_dir.glob("*.srt"):
        txt_file = out_dir / (srt_file.stem + ".txt")
        lines_out = []

        with srt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if timer_pattern.search(line):
                    continue
                lines_out.append(line)

        with txt_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))

        print(f"[OK] {srt_file.name} → {txt_file.name}")


if __name__ == "__main__":
    convert_folder_srt_to_txt(
        Path(r"C:\Users\basti\Downloads\A&T"),
        Path(r"C:\Users\basti\Downloads\A&T\txt")
    )