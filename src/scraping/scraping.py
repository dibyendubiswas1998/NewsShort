import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq




class Scrapper:
    def __init__(self) -> None:
        pass


    def global_news_scrapping(self, url="https://www.ndtv.com/topic/geopolitical", num_articles=5):
        try:
            news_list = [] # define empty list that stores all news
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            
            news_headlines = soup.find_all("div", class_="src_lst-rhs")

            for i, headlines in enumerate(news_headlines):
                if i >= num_articles:
                    break  # Stop after scraping the desired number of articles


                link = headlines.find("a")
                title = link.text.strip()
                href = link["href"]
                

                # move to that page based on link:
                linked_page = requests.get(href)
                linked_soup = BeautifulSoup(linked_page.content, "html.parser")

                # Find the Headline:
                all_paragraphs  = linked_soup.find_all("p", class_="")

                # Create a dictionary that containing information about current geopolitical news
                dct = {
                    'title': title,
                    'paragraph': ", ".join([paragraph.text.strip() for paragraph in all_paragraphs])
                }    
                news_list.append(dct)

            return news_list

        except Exception as ex:
            raise ex
    


    def domestic_news_scrapping(self, url="https://www.ndtv.com/india", num_articles=5):
        try:
            news_list = [] # define empty list that stores all news
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            
            news_headlines = soup.find_all("div", class_="news_Itm-cont")

            for i, headlines in enumerate(news_headlines):
                if i >= num_articles:
                    break  # Stop after scraping the desired number of articles
                
                link = headlines.find("a")
                title = link.text.strip()
                href = link["href"]
                
                # print(title)
                # print(link)

                # move to that page based on link:
                linked_page = requests.get(href)
                linked_soup = BeautifulSoup(linked_page.content, "html.parser")

                # Find the Headline:
                all_paragraphs  = linked_soup.find_all("p", class_="")

                # Create a dictionary that containing information about current geopolitical news
                dct = {
                    'title': title,
                    'paragraph': ", ".join([paragraph.text.strip() for paragraph in all_paragraphs])
                }    
                news_list.append(dct)

            return news_list

        except Exception as ex:
            raise ex



if __name__ == "__main__":
    scr = Scrapper()
    news = scr.domestic_news_scrapping()
    for dct in news:
        print("Titles: \t",  dct['title'])
        print("paragraph: \t", dct['paragraph'], '\n\n\n')
        


