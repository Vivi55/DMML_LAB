import requests
from bs4 import BeautifulSoup

res_foods = requests.get('http://www.xiachufang.com/explore/')
bs_foods = BeautifulSoup(res_foods.text,'html.parser')

tag_name = bs_foods.find_all('p',class_='name')
tag_ingredients = bs_foods.find_all('p',class_='ing ellipsis')
list_all = []
for x in range(len(tag_name)):
    list_food = [tag_name[x].text[18:-14],tag_name[x].find('a')['href'],tag_ingredients[x].text[1:-1]]
    list_all.append(list_food)
print(list_all)



def method_two():
    list_foods = bs_foods.find_all('div',class_='info pure-u')
    list_all = []
    for food in list_foods:

        tag_a = food.find('a')
        name = tag_a.text.strip()
        URL = 'http://www.xiachufang.com'+tag_a['href']
        tag_p = food.find('p',class_='ing ellipsis')
        ingredients = tag_p.text.strip()
        list_all.append([name,URL,ingredients])

    print(list_all)



