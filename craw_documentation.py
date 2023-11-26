from IPython.display import clear_output

urls = "https://ucy-linc-lab.github.io/fogify/"

from langchain.document_transformers import Html2TextTransformer
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import AsyncHtmlLoader

def crawlulr(start_url):
  url_contents=[]
  # Send a GET request to the URL
  response = requests.get(start_url)

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the HTML content
      soup = BeautifulSoup(response.text, 'html.parser')

      # Find all 'a' tags (links) in the page
      links = soup.find_all('a')

      # Extract and print all URLs
      for link in links:
          # Get the href attribute of each 'a' tag
          href = link.get('href')

          # Join the URL if it's relative
          full_url = urljoin(start_url, href)

          print(f"Crawling url: {full_url}")
          loader = AsyncHtmlLoader(full_url)
          docs = loader.load()
          html2text = Html2TextTransformer()
          docs_transformed = html2text.transform_documents(docs)

          url_contents.append(docs_transformed)
  else:
      print("Failed to retrieve the page")
  with open('/data/url_content.txt', 'w') as fp:
    for item in url_contents:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')
crawlulr(urls)
