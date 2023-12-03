from utils import convert_tsv_csv, relative_path

convert_tsv_csv(
    relative_path("data/english-to-spanish.tsv"),
    relative_path("data/english_to_spanish.csv"),
)
