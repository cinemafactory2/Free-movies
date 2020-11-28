from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import os
import pandas as pd

def printOpen(f_out, data = None): #open file, if data is given first write data then open
    if data:
        f = open(f_out, 'w')
        f.write(str(data))
        f.close()
    try:
        os.startfile(f_out)
    except Exception as e:
        print(e.__class__)

    return 

def write_f(f,*data):
    n = len(data)
    for i in range(n):
        f.write(data[i].replace(',','|')) #we replace ',' with '|' if string contain commas(in case)
        if i != n-1:
            f.write(',') #to change the column
        else:
            f.write('\n') #to change the row
    return

f_test = 'output.txt' #files that contain output of program for test run
_file = 'E:\\DoNotTouch\\Python\\projects\\Web Scrapping\\learning\\web_scrap.csv'
f = open(_file, 'w')

f.write("TITLE, PROUCT NAME, SHIPPING \n") # column for file

url = 'https://www.newegg.com/p/pl?Submit=StoreIM&Depa=1&Category=38'
uClient = uReq(url)
page = uClient.read()
uClient.close()

page_html = soup(page, 'html.parser')

containers = page_html.findAll('div',{'class':'item-container'})

for container in containers:   

    title = container.findAll('div', {'class' : 'item-branding'})
    title = title[0].a.img['title']

    product_name = container.findAll('a', {'class': 'item-title'})
    product_name = product_name[0].text 
    #scince it contains "," we replace ',' with '|'
    # because commas will change the colummn when we write in the CSV file
    product_name = product_name.replace(',', '|')

    shipping = container.findAll('li', {'class' : 'price-ship'})
    shipping = shipping[0].text.strip()

    write_f(f,title,product_name,shipping)

f.close()

printOpen(_file)
