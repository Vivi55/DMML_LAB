# coding=gbk
import requests
import csv
from bs4 import BeautifulSoup
"""# ʹ��headers��һ��Ĭ�ϵ�ϰ�ߣ�Ĭ�����Ѿ�������~
headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
# ��resquestģ�鷢�����󣬽���Ӧ�Ľ����ֵ������res��
url='https://www.zhihu.com/people/zhang-jia-wei/posts?page=1'
res=requests.get(url,headers=headers)
# ���״̬��
print(res.status_code)
# ��ӡ��ҳԴ����
#print(res.text)
# ��bs���н���
bstitle=BeautifulSoup(res.text,'html.parser')
# ��ȡ������Ҫ�ı�ǩ�����������
title=bstitle.find_all(class_='ContentItem-title')
# ��ӡtitle
print(title)
"""
# ����open()������csv�ļ�������������ļ�����articles.csv����д��ģʽ��w����newline=''��
csv_file=open('articles.csv','w',newline='',encoding='utf-8')
writer = csv.writer(csv_file)
# ����һ���б�
list2=['����','����','ժҪ']
# ����writer�����writerow()������������csv�ļ���д��һ������ �����⡱�͡����ӡ���"ժҪ"��
writer.writerow(list2)
headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
url='https://www.zhihu.com/api/v4/members/zhang-jia-wei/articles?'
articlelist=[]
offset=10
while True:
# ��װ����
    params={
        'include':'data[*].comment_count,suggest_edit,is_normal,thumbnail_extra_info,thumbnail,can_comment,comment_permission,admin_closed_comment,content,voteup_count,created,updated,upvoted_followees,voting,review_info,is_labeled,label_info;data[*].author.badge[?(type=best_answerer)].topics',
        'offset':'10',
        'limit':'10',
        'sort_by':'voteups',
        }
    # �������󣬲�����Ӧ���ݸ�ֵ������res����
    res=requests.get(url,headers=headers,params=params)
    # ȷ������ɹ�
    print(res.status_code)
    if int(res.status_code)==200:
        articles=res.json()
        data= articles['data']

        for i in data:
            list1=[i['title'],i['url'],i['excerpt']]
            articlelist.append(list1)
        offset=offset+20
        if offset>30:
            break
        # ���offset����30����������ҳ����ֹͣ
        # ��������������һ��˼·ʵ�֡�������������������������������
        # �����is_end����Ӧ��ֵ��True���ͽ���whileѭ����
        # if articles['paging']['is_end'] == True:
        # break
        # ������������������������������������������������������������������������
# ��ӡ����
csv_file.close()
print('okay')