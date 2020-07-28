import random

list_food = ['KFC', 'Cheese','Egg','Butter','Margarine','Yogurt','Cottage cheese','Ice cream','Cream','Sandwich','Sausage','Hamburger','Hot dog']
list_choice = []

def decision(list):
    while True:
        food= random.choice(list)
        judgement=input('How about the[%s]? y or n')
        if judgement=='y':
            print("Good idea! Let us to have [%s]"%(food))
            break

reason =input("Why you have no idea about eating? 1.totally no clue. 2. have several choice, do not know which one is better, answer 1 or 2\n")
if reason==1:
    decision(list_food)
elif reason==2:
    add =True
    while add:
        choice= input('what makes u confused? print one by one and end with y:')
        if choice!='y':
            list_choice.append(choice)
        if choice=='y':
            add=False
    decision(list_choice)
else:
    print("mistake")


list_food = ['KFC', '蒸菜馆', '楼下快餐店', '桂林米粉', '东北饺子', '金牌猪脚饭', '三及第汤饭']  # 备选菜单，可自定义。
list_choice = []

# 由于两个原因都包含判断过程，所以，为了让代码更简洁，可将其封装成函数。
def choose(list):
    while True:
        food = random.choice(list)
        judgement = input('去吃【%s】好不好啊？同意的话输入y，不想吃直接回车即可。'%(food))
        if judgement == 'y':
            print('去吃【%s】！就这么愉快地决定啦！'%(food))
            break

# 判断环节
reason = int(input('你不知道吃什么的原因是：1.完全不知道吃什么；2.在几家店之间徘徊（请输入1或2）：'))
if reason == 1:
    choose(list_food)
elif reason == 2:
    add = True
    while add:
        choice = input('请输入让你犹豫的店名（注：一家一家输，完成后输入y）：')
        if choice != 'y':  # 这个判断语句，是为了不将 y 也添加到菜单里。
            list_choice.append(choice)
        if choice == 'y':
            add = False
    choose(list_choice)
else:
    print('抱歉，目前还不支持第三种情况——不过，你可以加代码哦。')




