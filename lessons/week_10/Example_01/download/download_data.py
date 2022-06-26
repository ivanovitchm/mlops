"""
Creator: Ivanovitch Silva
Date: 29 Jan. 2022
Download the raw data
"""
import argparse
import logging
import pathlib
import wandb
import requests
import tempfile

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


def process_args(args):
    # Download file, streaming so we can download files larger than
    # the available memory. We use a named temporary file that gets
    # destroyed at the end of the context, so we don't leave anything
    # behind and the file gets removed even in case of errors
    logger.info(f"Downloading {args.file_url} ...")
    with tempfile.NamedTemporaryFile(mode='wb+') as fp:

        logger.info("Creating run")
        with wandb.init(job_type="download_data") as run:
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

            # Make sure the file has been written to disk before uploading
            # to W&B
            fp.flush()

            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={'original_url': args.file_url}
            )
            artifact.add_file(fp.name, name=args.artifact_name)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B", fromfile_prefix_chars="@"
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    ARGS = parser.parse_args()

    process_args(ARGS)
