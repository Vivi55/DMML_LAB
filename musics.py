# coding=gbk
import requests, openpyxl

singer=input('who is your favourite singer?')
wb = openpyxl.Workbook()
sheet = wb.active
sheet.title = 'song'
sheet['A1'] = '������'  # �ӱ�ͷ����A1��Ԫ��ֵ
sheet['B1'] = '����ר��'  # �ӱ�ͷ����B1��Ԫ��ֵ
sheet['C1'] = '����ʱ��'  # �ӱ�ͷ����C1��Ԫ��ֵ
sheet['D1'] = '��������'  # �ӱ�ͷ����D1��Ԫ��ֵ
url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'
for x in range(5):

    params = {
        'ct': '24',
        'qqmusic_ver': '1298',
        'new_json': '1',
        'remoteplace': 'sizer.yqq.song_next',
        'searchid': '64405487069162918',
        't': '0',
        'aggr': '1',
        'cr': '1',
        'catZhida': '1',
        'lossless': '0',
        'flag_qc': '0',
        'p': str(x + 1),
        'n': '20',
        'w': singer,
        'g_tk': '5381',
        'loginUin': '0',
        'hostUin': '0',
        'format': 'json',
        'inCharset': 'utf8',
        'outCharset': 'utf-8',
        'notice': '0',
        'platform': 'yqq.json',
        'needNewCode': '0'
    }

    res_music = requests.get(url, params=params)
    json_music = res_music.json()
    list_music = json_music['data']['song']['list']
    for music in list_music:
        print(music['name'])
        print('����ר����' + music['album']['name'])
        print('����ʱ����' + str(music['interval']) + '��')
        print('�������ӣ�https://y.qq.com/n/yqq/song/' + music['file']['media_mid'] + '.html\n\n')

wb.save('Jay.xlsx')