# coding=gbk
import requests
# ����requestsģ��
for i in range(5):
    res_comments = requests.get('https://c.y.qq.com/base/fcgi-bin/fcg_global_comment_h5.fcg?g_tk=5381&loginUin=0&hostUin=0&format=json&inCharset=utf8&outCharset=GB2312&notice=0&platform=yqq.json&needNewCode=0&cid=205360772&reqtype=2&biztype=1&topid=102065756&cmd=6&needmusiccrit=0&pagenum='+str(i)+'&pagesize=15&lasthotcommentid=song_102065756_3202544866_44059185&domain=qq.com&ct=24&cv=10101010')
    # ����get���������������б�
    json_comments = res_comments.json()
    # ʹ��json()��������response����תΪ�б�/�ֵ�
    list_comments = json_comments['comment']['commentlist']
    # һ��һ���ȡ�ֵ䣬��ȡ�����б�
    for comment in list_comments:
    # list_comments��һ���б�comment���������Ԫ��
        print(comment['rootcommentcontent'])
        # �������
        print('-----------------------------------')
        # ����ͬ�����۷ָ�����


        """��ȡ5ҳ�赥
         res_music = requests.get(url, params=params)
    # ����get��������������ֵ�
    json_music = res_music.json()
    # ʹ��json()��������response����תΪ�б�/�ֵ�
    list_music = json_music['data']['song']['list']
    # һ��һ���ȡ�ֵ䣬��ȡ�赥�б�
    for music in list_music:
        # list_music��һ���б�music���������Ԫ��
        print(music['name'])
        # ��nameΪ�������Ҹ�����
        print('����ר����' + music['album']['name'])
        # ����ר����
        print('����ʱ����' + str(music['interval']) + '��')
        # ���Ҳ���ʱ��
        print('�������ӣ�https://y.qq.com/n/yqq/song/' + music['mid'] + '.html\n\n')
        # ���Ҳ�������
        """
    """ ��ȡϲ�����ֵĸ赥
    import requests
#����requestsģ��
singer = input('����ϲ���ĸ�����˭ѽ��')
url = 'https://c.y.qq.com/soso/fcgi-bin/client_search_cp'
for x in range(5):
    
    params = {
    'ct':'24',
    'qqmusic_ver': '1298',
    'new_json':'1',
    'remoteplace':'txt.yqq.song',
    'searchid':'70717568573156220',
    't':'0',
    'aggr':'1',
    'cr':'1',
    'catZhida':'1',
    'lossless':'0',
    'flag_qc':'0',
    'p':str(x+1),
    'n':'20',
    'w':singer,
    'g_tk':'714057807',
    'loginUin':'0',
    'hostUin':'0',
    'format':'json',
    'inCharset':'utf8',
    'outCharset':'utf-8',
    'notice':'0',
    'platform':'yqq.json',
    'needNewCode':'0'    
    }
    # ��������װΪ�ֵ�
    res_music = requests.get(url,params=params)
    # ����get��������������б�
    json_music = res_music.json()
    # ʹ��json()��������response����תΪ�б�/�ֵ�
    list_music = json_music['data']['song']['list']
    # һ��һ���ȡ�ֵ䣬��ȡ�赥�б�
    for music in list_music:
    # list_music��һ���б�music���������Ԫ��
        print(music['name'])
        # ��nameΪ�������Ҹ�����
        print('����ר����'+music['album']['name'])
        # ����ר����
        print('����ʱ����'+str(music['interval'])+'��')
        # ���Ҳ���ʱ��
        print('�������ӣ�https://y.qq.com/n/yqq/song/'+music['mid']+'.html\n\n')
        # ���Ҳ�������
    """