import requests
import random
from bs4 import BeautifulSoup as soup

def suggestMovie():
    Movies = page.select("td.titleColumn")
    
    Title = [movie.a.text for movie in Movies]
    Actors = [movie.a['title'] for movie in Movies]
    Year = [movie.span.text for movie in Movies]
    Link = ['https://www.imdb.com/' + movie.a['href'] for movie in Movies]

    Ratings = page.findAll(class_ = 'ratingColumn imdbRating')
    Rating = [rating.strong.text for rating in Ratings]

    Total_movies = len(Title)
    while True:
        idx = random.randrange(0,Total_movies)

        print('HERE IS THE ONE OF THE BEST MOVIE FOR YOU...!')
        print(f"Movie : {Title[idx]}")
        print(f"Year : {Year[idx]}")
        print(f"Actor : {Actors[idx]}")
        print(f"Rating : {Rating[idx]}")
        print(f"Clik on link to know more about movie : {Link[idx]}")

        cont_ = input("Do you want another movie (y/n)?")
        if cont_ != 'y':
            break

if __name__ == "__main__":
    url = "https://www.imdb.com/chart/top/"
    response = requests.get(url)
    page_html = response.text
    page = soup(page_html, 'html.parser')

    suggestMovie()
    
    
    
