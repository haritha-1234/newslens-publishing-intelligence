# Shared utility functions used across all modules.
# Keeping these in one place avoids code duplication
# and makes it easy to update cleaning logic everywhere.

import re


def clean_title(text):
    """Clean a news article title for NLP processing."""

    # convert to lowercase
    text = str(text).lower()

    # remove anything that is not a letter or space
    text = re.sub(r"[^a-z\s]", "", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


CATEGORY_MAP = {
    "b": "Business",
    "t": "Science & Tech",
    "e": "Entertainment",
    "m": "Health"
}