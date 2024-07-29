import pandas as pandas
import glob
import os


import sys
import logging
import logging.handlers

def setup_logger(log_file_path):
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the logging level for the file handler

    # Create a console handler to log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Set the logging level for the console handler

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
if __name__ == '__main__':
    log_file_path = './logs/remove_no_mutations.log'
    logger = setup_logger(log_file_path)
    logger.info("Starting remove_no_mutations.py")

    directory = os.getcwd()+'/cache_dir/'
    naming_format = '*_sub.csv'  # This will match any file that ends with .json

    files = glob.glob(f'{directory}/{naming_format}')

    total_entries = 0
    num_entries = 0


    for file in files:
        print(file)
        df = pandas.read_csv(file)
        ##lop through the rows and find if the mutation column is not None
        found = False
        for index, row in df.iterrows():
            if row['change'] != '-':
                found = True
                break
        if not found:
            # delete the file
            print(f"Deleting {file}")
            os.remove(file)
        else:
            total_entries += df.shape[0]
            num_entries += 1
            

    print(f"Total entries: {total_entries}")
    print(f"Total files: {len(files)}")
    print(f"Average entries per file: {total_entries/num_entries}")
    print(f"Total files deleted: {len(files) - num_entries}")
    print(f"Remaining files: {num_entries}")
    logger.info(f"Total entries: {total_entries}")
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Average entries per file: {total_entries/num_entries}")
    logger.info(f"Total files deleted: {len(files) - num_entries}")
    logger.info(f"Remaining files: {num_entries}")

    logger.info("Finished remove_no_mutations.py")

