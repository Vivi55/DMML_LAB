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
    # ��nameΪ�������Ҹ�����
    print('����ר����' + music['album']['name'])
    # ����ר����
    print('����ʱ����' + str(music['interval']) + '��')
    # ���Ҳ���ʱ��
    print('�������ӣ�https://y.qq.com/n/yqq/song/' + music['mid'] + '.html\n\n')

def typ():

    # ����jsonģ��
    a = [1,2,3,4]
    # ����һ���б�a��
    b = json.dumps(a)
    # ʹ��dumps()���������б�aת��Ϊjson��ʽ���ַ�������ֵ��b��
    print(b)
    # ��ӡb��
    print(type(b))
    # ��ӡb���������͡�
    c = json.loads(b)
    # ʹ��loads()��������json��ʽ���ַ���bתΪ�б���ֵ��c��
    print(c)
    # ��ӡc��
    print(type(c))
    # ��ӡc����������