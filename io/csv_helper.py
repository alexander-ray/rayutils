import csv


# Meant to be used if you fuck up, need to copy output to a text file
def file_of_lists_to_csv(old_filename, new_filename):
    """
    Helper function to convert lists from text files to CSVs.
    Only use when you forget to write to a CSV from jupyter notebooks.

    :param old_filename: Path of text file
    :param new_filename: Intended path of CSV
    """
    with open(old_filename, 'r') as f_old:
        lines = (line.rstrip() for line in f_old)  # All lines including the blank ones
        lines = (eval(line) for line in lines if line)  # Non-blank lines

        with open(new_filename, 'a') as f_new:
            writer = csv.writer(f_new)
            for line in lines:
                writer.writerow(line)