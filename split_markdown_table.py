from pathlib import Path


INPUT_FILE = Path("tableConvert.com_yk0bx9.txt")
OUTPUT_DIR = Path("out")
ROWS_PER_FILE = 10_000
TITLE = "Список камер ЕСВМ (Единая система видеомониторинга) г. Алматы"


def write_part(part_number: int, table_header: list[str], rows: list[str]) -> None:
    output_path = OUTPUT_DIR / f"cameras_part_{part_number:03}.md"
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(TITLE + "\n\n")
        output_file.writelines(table_header)
        output_file.writelines(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as input_file:
        table_header = [input_file.readline(), input_file.readline()]

        if not all(table_header):
            raise ValueError("Input file must contain a markdown table header.")

        part_number = 1
        rows: list[str] = []

        for line in input_file:
            rows.append(line)

            if len(rows) == ROWS_PER_FILE:
                write_part(part_number, table_header, rows)
                part_number += 1
                rows = []

        if rows:
            write_part(part_number, table_header, rows)


if __name__ == "__main__":
    main()
