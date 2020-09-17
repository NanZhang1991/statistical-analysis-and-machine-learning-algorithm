# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:16:35 2020

@author: YJ001
"""
"""
频繁项集：
频繁项集是指那些经常出现在一起的物品，例如下表的{葡萄酒、尿布、豆奶}，从上面的数据集中也可以找到尿布->葡萄酒的关联规则，这意味着有人买了尿布，那很有可能他也会购买葡萄酒。那如何定义和表示频繁项集和关联规则呢？这里引入支持度和可信度（置信度）。

支持度：
支持度：一个项集的支持度被定义为数据集中包含该项集的记录所占的比例，上图中，豆奶的支持度为4/5，（豆奶、尿布）为3/5。支持度是针对项集来说的，因此可以定义一个最小支持度，只保留最小支持度的项集。

置信度：
可信度（置信度）：针对如{尿布}->{葡萄酒}这样的关联规则来定义的。计算为 支持度{尿布，葡萄酒}/支持度{尿布}，其中{尿布，葡萄酒}的支持度为3/5，{尿布}的支持度为4/5，所以“尿布->葡萄酒”的可行度为3/4=0.75，这意味着尿布的记录中，我们的规则有75%都适用（买了尿布的顾客有75%还会买葡萄酒）。
"""
 

import pandas as pd


shopping_list = [['豆奶','莴苣'],
	        ['莴苣','尿布','葡萄酒','甜菜'],
	        ['豆奶','尿布','葡萄酒','橙汁'],
	        ['莴苣','豆奶','尿布','葡萄酒'],
	        ['莴苣','豆奶','尿布','橙汁']]
 
shopping_df = pd.DataFrame(shopping_list)


def deal(data):
	return data.dropna().tolist()
df_arr = shopping_df.apply(deal,axis=1).tolist()	


"""由于mlxtend的模型只接受特定的数据格式。（TransactionEncoder类似于独热编码，每个值转换为一个唯一的bool值）"""
from mlxtend.preprocessing import TransactionEncoder	# 传入模型的数据需要满足特定的格式，可以用这种方法来转换为bool值，也可以用函数转换为0、1
 
te = TransactionEncoder()	# 定义模型
df_tf = te.fit_transform(df_arr)
# df_01 = df_tf.astype('int')			# 将 True、False 转换为 0、1 # 官方给的其它方法
# df_name = te.inverse_transform(df_tf)		# 将编码值再次转化为原来的商品名
df = pd.DataFrame(df_tf,columns=te.columns_)

"""求频繁项集：

导入apriori方法设置最小支持度min_support=0.05求频繁项集，还能选择出长度大于x的频繁项集。
"""
from mlxtend.frequent_patterns import apriori
 
frequent_itemsets = apriori(df,min_support=0.05,use_colnames=True)	# use_colnames=True表示使用元素名字，默认的False使用列名代表元素
# frequent_itemsets = apriori(df,min_support=0.05)
frequent_itemsets.sort_values(by='support',ascending=False,inplace=True)	# 频繁项集可以按支持度排序
print('求频繁项集')
print(frequent_itemsets[frequent_itemsets.itemsets.apply(lambda x: len(x)) >= 2])  # 选择长度 >=2 的频繁项集
print()
"""
求关联规则：

导入association_rules方法判断'confidence'大于0.9，求关联规则。
"""

from mlxtend.frequent_patterns import association_rules
 
association_rule = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.9)	# metric可以有很多的度量选项，返回的表列名都可以作为参数
association_rule.sort_values(by='leverage',ascending=False,inplace=True)    #关联规则可以按leverage排序
print('关联规则')
print(association_rule)
"""
antecedents：规则先导项

consequents：规则后继项

antecedent support：规则先导项支持度

consequent support：规则后继项支持度

support：规则支持度 （前项后项并集的支持度）

confidence：规则置信度 （规则置信度：规则支持度support / 规则先导项）

lift：规则提升度，表示含有先导项条件下同时含有后继项的概率，与后继项总体发生的概率之比。

leverage：规则杠杆率，表示当先导项与后继项独立分布时，先导项与后继项一起出现的次数比预期多多少。

conviction：规则确信度，与提升度类似，但用差值表示。
"""

