import requests
from bs4 import BeautifulSoup as soup
import pandas as pd

url = 'https://www.newegg.com/p/pl?Submit=StoreIM&Depa=1&Category=38'
uClient = requests.get(url)
page = uClient.content
uClient.close()

page_html = soup(page, 'html.parser')

container = page_html.find(class_='items-view is-grid')  

Title = [item.img['title'] for item in container.select('.item-container .item-brand')]
Product_Name = [item.text for item in container.select('.item-container .item-title')]
Shipping = [item.text.strip() for item in container.select('.item-container .price-ship')]

Data = pd.DataFrame({
    'TITLE': Title,
    'Product_Name': Product_Name,
    'Shipping': Shipping,
})

Data = Data.sort_values('TITLE')
Data.to_csv('web_scrap.csv', index = False)



