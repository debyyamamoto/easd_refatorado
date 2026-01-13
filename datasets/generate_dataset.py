import argparse
from parquet_dataset import ParquetDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet Dataset")
    parser.add_argument("filepath", type=str, help="Caminho para o arquivo do dataset - ex: datasets/Mixed/German.csv")
    parser.add_argument(
        "-sep",
        "--separator",
        type=str,
        default=",",
        help="Caso o arquivo do dataset use um separador, especifique para fazer a leitura corretamente. - Ex: ',', ';', ' ' etc",
    )
    parser.add_argument(
        "json_schema", type=str, help="Caminho para o arquivo do JSON Schema para o datatype dos atributos"
    )
    parser.add_argument("-n", "--name", type=str, default="", help="Nomear arquivo parquet")

    args = parser.parse_args()
    dataset_build = ParquetDataset(args.name)
    dataset_build.generate(args.filepath, args.separator, args.json_schema)
