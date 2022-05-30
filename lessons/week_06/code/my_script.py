"""
Creator: Ivanovitch Silva
Date: 18 Nov. 2021
A Brief Refresher on Python Scripting with Argparse
"""
import argparse
import logging

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def process_args(args):
    """
    Get the args and show using logs
    Param
        args - command line arguments
    """
    logger.info("This is %s", args.artifact_name)
    logger.info("This is %s",args.optional_arg)


if __name__ == "__main__":
    # This block is executed only if this file is being
    # executed as a script. It is NOT executed if the file
    # is imported as a module
    parser = argparse.ArgumentParser(
        description="This is a tutorial on argparse")

    # add the argument artifact_name
    parser.add_argument("--artifact_name",
                        type=str,
                        help="Name and version of artifact",
                        required=True)

    # add the argument optional_arg
    parser.add_argument("--optional_arg",
                        type=float,
                        help="An optional argument",
                        required=False,
                        default=2.3)
    # get arguments
    ARGS = parser.parse_args()

    # show arguments
    process_args(ARGS)
    