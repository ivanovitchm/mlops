"""Recent coding books module
"""


def find_recent_coding_books(recent_books_path, coding_books_path):
    """Find the recent coding books in all library

    Arguments
    recent_books_path: (str). Path of recent books
    coding_books_path: (str). Path for all coding books

    Returns
    recent_coding_books: (set). A set of recent coding books

    """
    with open(recent_books_path) as file1:
        recent_books = file1.read().split('\n')

    with open(coding_books_path) as file2:
        coding_books = file2.read().split('\n')

    recent_coding_books = set(recent_books).intersection(coding_books)
    return recent_coding_books


if __name__ == "__main__":
    result = find_recent_coding_books(
        'books_published_last_two_years.txt', 'all_coding_books.txt')
    print(result)
