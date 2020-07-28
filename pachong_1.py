# coding=gbk
import requests
import json
from bs4 import BeautifulSoup
url='https://y.qq.com/portal/search.html#page=1&searchid=1&remoteplace=txt.yqq.top&t=song&w=TROYE%20SIVAN'
json_music=requests.get(url).json()
#bs_music = BeautifulSoup(res_music.text,'html.parser')

list_music = json_music['data']['song']['list']
for music in list_music:
    print(music['name'])
    # 以name为键，查找歌曲名
    print('所属专辑：' + music['album']['name'])
    # 查找专辑名
    print('播放时长：' + str(music['interval']) + '秒')
    # 查找播放时长
    print('播放链接：https://y.qq.com/n/yqq/song/' + music['mid'] + '.html\n\n')

def typ():

    # 引入json模块
    a = [1,2,3,4]
    # 创建一个列表a。
    b = json.dumps(a)
    # 使用dumps()函数，将列表a转换为json格式的字符串，赋值给b。
    print(b)
    # 打印b。
    print(type(b))
    # 打印b的数据类型。
    c = json.loads(b)
    # 使用loads()函数，将json格式的字符串b转为列表，赋值给c。
    print(c)
    # 打印c。
    print(type(c))
    # 打印c的数据类型