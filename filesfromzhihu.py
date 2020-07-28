# coding=gbk
import requests
import csv
from bs4 import BeautifulSoup
"""# 使用headers是一种默认的习惯，默认你已经掌握啦~
headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
# 用resquest模块发起请求，将响应的结果赋值给变量res。
url='https://www.zhihu.com/people/zhang-jia-wei/posts?page=1'
res=requests.get(url,headers=headers)
# 检查状态码
print(res.status_code)
# 打印网页源代码
#print(res.text)
# 用bs进行解析
bstitle=BeautifulSoup(res.text,'html.parser')
# 提取我们想要的标签和里面的内容
title=bstitle.find_all(class_='ContentItem-title')
# 打印title
print(title)
"""
# 调用open()函数打开csv文件，传入参数：文件名“articles.csv”、写入模式“w”、newline=''。
csv_file=open('articles.csv','w',newline='',encoding='utf-8')
writer = csv.writer(csv_file)
# 创建一个列表
list2=['标题','链接','摘要']
# 调用writer对象的writerow()方法，可以在csv文件里写入一行文字 “标题”和“链接”和"摘要"。
writer.writerow(list2)
headers={'user-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
url='https://www.zhihu.com/api/v4/members/zhang-jia-wei/articles?'
articlelist=[]
offset=10
while True:
# 封装参数
    params={
        'include':'data[*].comment_count,suggest_edit,is_normal,thumbnail_extra_info,thumbnail,can_comment,comment_permission,admin_closed_comment,content,voteup_count,created,updated,upvoted_followees,voting,review_info,is_labeled,label_info;data[*].author.badge[?(type=best_answerer)].topics',
        'offset':'10',
        'limit':'10',
        'sort_by':'voteups',
        }
    # 发送请求，并把响应内容赋值到变量res里面
    res=requests.get(url,headers=headers,params=params)
    # 确认请求成功
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
        # 如果offset大于30，即爬了两页，就停止
        # ——————另一种思路实现————————————————
        # 如果键is_end所对应的值是True，就结束while循环。
        # if articles['paging']['is_end'] == True:
        # break
        # ————————————————————————————————————
# 打印看看
csv_file.close()
print('okay')