"""
This is a container for custom exceptions
"""


class FormatException(Exception):
    def __init__(self, format, expected_format):
        self.format = format
        self.expected_format = expected_format


class ContradictoryException(Exception):
    def __init__(self, message: str, *vals):
        self.vals = vals
        self.message = message

    def __str__(self):
        return self.message + ' ' + str(self.vals)


if __name__ == '__main__':
    val1 = 'n_filter'
    val2 = 's_filter'
    raise ContradictoryException("Invalid configuration at:", val1, val2)