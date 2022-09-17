import re

def string_cleaner(input: str) -> str:
    """
    Function to clean up strings.

    Args:
        input (str): String to be cleaned.

    Returns:
        str: Cleaned string.
    """
    
    # turning lowercase
    input = input.lower()

    input = re.sub(r'\-', ' ', input)

    # removing punctuation and other non-alphanumeric characters
    input = re.sub(r'[^\w\s]', '', input)
    
    return input

