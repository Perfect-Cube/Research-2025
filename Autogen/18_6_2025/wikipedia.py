#pip install wikipedia
import wikipedia

def get_wikipedia_summary(search_term, sentences=5):
  """
  Fetches a summary from Wikipedia for a given search term.

  Args:
    search_term: The term to search for on Wikipedia.
    sentences: The number of sentences to include in the summary.  Defaults to 5.

  Returns:
    A string containing the summary, or None if an error occurred.
  """
  try:
    # Search Wikipedia for the given term
    page = wikipedia.page(search_term, auto_suggest=False, sentences=sentences)  # auto_suggest=False to prevent unintended suggestions

    # Get the summary (the first paragraph)
    summary = wikipedia.summary(search_term, sentences=sentences) # Gets the summary of the page.

    return summary
  except wikipedia.exceptions.PageError:
    print(f"Error: Page '{search_term}' not found on Wikipedia.")
    return None
  except wikipedia.exceptions.DisambiguationError as e:
    print(f"Error: Disambiguation error for '{search_term}'.  Possible options:")
    print(e.options)
    return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None



def get_wikipedia_data(search_term, aspects=["summary", "key points"]):
    """
    Fetches data from Wikipedia for a given search term, including summary and other aspects.

    Args:
        search_term: The term to search for on Wikipedia.
        aspects: A list of aspects to extract. Defaults to ["summary", "key points"].
    Returns:
        A dictionary containing the extracted data, or None if an error occurred.
    """
    data = {}
    try:
        page = wikipedia.page(search_term, auto_suggest=False)
        data['title'] = page.title
        data['summary'] = wikipedia.summary(search_term, sentences=5)

        if "key points" in aspects:
            # Extract key points (This can be tricky, as it's not a direct property)
            # This is a basic attempt; more sophisticated parsing might be needed.
            data['key_points'] = extract_key_points(page.content)

        return data
    except wikipedia.exceptions.PageError:
        print(f"Error: Page '{search_term}' not found on Wikipedia.")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Error: Disambiguation error for '{search_term}'.  Possible options:")
        print(e.options)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def extract_key_points(text):
    """
    A very basic attempt to extract key points from Wikipedia text.
    This is highly dependent on the structure of the Wikipedia article.
    More sophisticated methods (e.g., NLP techniques) may be needed for 
    more reliable extraction.

    Args:
        text: The Wikipedia article text.

    Returns:
        A list of key points (sentences).
    """
    sentences = text.split('.')
    key_points = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # Ignore empty sentences
            key_points.append(sentence)
    return key_points[:3] #Return the first three as key points.



if __name__ == "__main__":
  # Example Usage
  search_term = "Python programming language"

  # Get a summary
  summary = get_wikipedia_summary(search_term)
  if summary:
    print("Summary:")
    print(summary)

  # Get more detailed data
  data = get_wikipedia_data(search_term, aspects=["summary", "key points"])
  if data:
    print("\nData:")
    print(f"Title: {data['title']}")
    print(f"Summary: {data['summary']}")
    print("Key Points:")
    for point in data['key_points']:
        print(f"- {point}")

  # Example with a potentially ambiguous term
  ambiguous_term = "Java"
  ambiguous_data = get_wikipedia_data(ambiguous_term, aspects=["summary"]) # just get summary for the ambiguous term
  if ambiguous_data:
      print("\nData for ambiguous term:")
      print(ambiguous_data)
