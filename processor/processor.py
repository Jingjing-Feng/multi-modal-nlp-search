# Take in a folder, process all the files within the folder recursively.
# For each file, process it and save the embedding to elasticsearch.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import datetime
import rich
import warnings
from data.data import Document
from elasticsearch import Elasticsearch
from embedding_converter import (
    convert_to_pure_text,
    convert_audio_to_text,
    convert_image_to_text,
    convert_pdf_to_text,
    convert_video_to_text,
)
from model.sbert import SBertModel
from rich.progress import Progress

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

VALID_FILE_EXTENSIONS = [".pdf", ".mp3", ".txt", ".png", ".jpg", ".jpeg"]


# Setup elasticsearch
ES = Elasticsearch("http://localhost:9200")
INDEX_NAME = "nls"
INDEX_MAPPING = {
    "properties": {
        "filename": {"type": "text", "analyzer": "english"},
        "extension": {"type": "text"},
        "text": {"type": "text", "analyzer": "english"},
        "embedding": {
            "type": "dense_vector",
            "dims": SBertModel.get_dimension(),
            "index": True,
            "similarity": "cosine",
        },
    }
}


def process_file(file_path: str) -> bool:
    """
    Process a single file. Return True if the file is processed successfully, False otherwise.
    """

    if not file_path.lower().endswith(tuple(VALID_FILE_EXTENSIONS)):
        rich.print(
            f"[yellow]Skipping file {file_path} because it is not a valid file type.[/yellow]"
        )
        return False

    rich.print(f"[green]Processing file {file_path}...[/green]")

    extension = os.path.splitext(file_path)[1].lower()
    file_name = file_path.split("/")[-1]

    if extension == ".txt":
        # process txt file
        text = convert_to_pure_text(file_path)
    elif extension == ".pdf":
        # process pdf file
        text = convert_pdf_to_text(file_path)
    elif extension == ".mp3":
        # process mp3 file
        text = convert_audio_to_text(file_path)
    elif extension in [".png", ".jpg", ".jpeg"]:
        # process image file
        text = convert_image_to_text(file_path)
    elif extension == ".mp4":
        # process video file
        text = convert_video_to_text(file_path)
    else:
        raise ValueError(f"Should not reach here.")

    document = Document(
        filename=file_name,
        text=text,
        extension=extension,
        created=datetime.datetime.fromtimestamp(os.path.getmtime(file_path)),
        size=os.path.getsize(file_path),
        path=file_path,
        embedding=SBertModel.get_embedding(file_name + "\n" + text),
    )

    # Ingest the processed document to elasticsearch
    index_file(document)

    ES.indices.refresh(index=INDEX_NAME)

    return True


def index_file(document: Document):
    ES.index(index=INDEX_NAME, body=document.to_index_format())


def get_files_in_folder(folder_path: str) -> list[str]:
    """
    Get all the files in the folder recursively.
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def process_files(folder_path: str):
    """
    Process all the files within the folder recursively.
    """
    skipped = []

    files = get_files_in_folder(folder_path)

    with Progress() as progress:
        task = progress.add_task("[green]Processing files...[/green]", total=len(files))

        for file in files:
            success = process_file(file)
            if not success:
                skipped.append(file)
            progress.update(task, advance=1)

    rich.print(f"[yellow]Skipped {len(skipped)} files.[/yellow]")
    rich.print(f"[green] Done![/green]")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--folder_path",
        required=True,
        type=str,
        help="The path to the folder to process",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing index",
    )
    args = argparser.parse_args()

    if args.overwrite:
        rich.print(f"[red bold]Overwriting the existing index...[/red bold]")
        if ES.indices.exists(index=INDEX_NAME):
            ES.indices.delete(index=INDEX_NAME)

    # Create the index
    if not ES.indices.exists(index=INDEX_NAME):
        ES.indices.create(index=INDEX_NAME, mappings=INDEX_MAPPING)

    rich.print(f"[green bold]Processing files in {args.folder_path}...[/green bold]")
    process_files(args.folder_path)
