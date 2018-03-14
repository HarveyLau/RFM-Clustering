# -*- coding: utf-8 -*-
# 使用K-Means算法聚类消费行为特征数据 Python

import pandas as pd
from sklearn.cluster import KMeans
import time
import multiprocessing

# 参数初始化
inputfile = '/Users/liuzhanheng/Desktop/030.xlsx'  # 销量及其他属性数据
outputfile = '/Users/liuzhanheng/Desktop/030_output.csv'  # 保存结果的文件名
iteration = 2000000000  # 聚类最大循环次数
W = [0.221, 0.341, 0.439]
H = [u'重要价值会员',u'潜力会员',u'重要深耕会员',u'新会员',u'重要唤回会员', u'一般维持会员', u'重要挽留会员',u'流失会员']
def comput(max,a,b,c):
    return (max-a)*W[0] + b*W[1] + c*W[2]
def comput_score(max,min,a,flag):
    if flag: 
       return (max - a) / (max - min)
    else:
       return (a - min) / (max - min)

def read_data(input_file_):
    data = pd.read_excel(input_file_, index_col='ID')  # 读取数据
    return data

def clear_data(data_):
    data_r = 1.0 * (data_['R'].max() - data_['R']) / (data_['R'].max() - data_['R'].min())
    data_f = 1.0 * (data_['F'] - data_['F'].min()) / (data_['F'].max() - data_['F'].min())
    data_m = 1.0 * (data_['M'] - data_['M'].min()) / (data_['M'].max() - data_['M'].min())
    data_zs = pd.concat([data_r, data_f, data_m], axis=1)
    r_max = data_['R'].max()
    f_max = data_['F'].max()
    m_max = data_['M'].max()
    r_mean = data_['R'][100:-100].mean()        #剔除最高，前100 降低平均值
    f_mean = data_['F'][100:-1].mean()
    m_mean = data_['M'][100:-350].mean()        #剔除最高，前350 降低平均值
    r_min = data_['R'].min()
    f_min = data_['F'].min()
    m_min = data_['M'].min()
    return data_zs, r_max,f_max,m_max, r_mean, f_mean, m_mean,r_min,f_min,m_min

def k_means(data_zs):
    model = KMeans(n_clusters=8, n_jobs=8, max_iter=iteration)  # 分为k类，并发数8
    model.fit(data_zs)  # 开始聚类
    return model

def clear_k_means(model):
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r3 = pd.Series(r2[0] * W[0] + r2[1] * W[1] + r2[2] * W[2])
    r = pd.concat([r2, r1, r3], axis=1)  # axis=1 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = [u'R_质心'] + [u'F_质心'] + [u'M_质心'] + [u'类别数目'] + [u'分数']  # 重命名表头
    r = r.sort_values(by=[u'分数'])
    s = pd.Series(pd.DataFrame(r).index)
    r4 = pd.Series([8, 7, 6, 5, 4, 3, 2, 1])
    s1 = pd.concat([s,r4],axis=1)
    s1.columns = [u'聚类类别'] + [u'排名']
    # rs = pd.concat([r, r4],axis=1)
    # rs.columns = [u'R_质心'] + [u'F_质心'] + [u'M_质心'] + [u'类别数目'] + [u'分数'] + [u'排名']
    return r,s1

def init_clear_data(data,model,s):
    rs = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细输出每个样本对应的类别
    rs.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
    #rs.merge(s, on = u'聚类类别',how = 'left')
    #rs[u'排名'] = rs[u'聚类类别'].map(s[u'排名'])
    rs = pd.merge(rs,s,how='left')
    #rs = rs.drop(u'聚类类别',1)
    #rs.columns = list(data.columns) + [u'聚类类别1']
    #loan_inner = pd.merge(loanstats, member_grade, how='inner')
    # rs.columns = list(data.columns) + [u'排名']
    print(s)
    print(rs.tail)
    rs[u'会员分类'] = None
    rs[u'会员价值分数'] = None
    rs[u'波动'] = None
    rs.sort_values(by='R')
    return rs

# 详细输出原始数据及其类别
def clear_s_data(data, model, r_max, f_max, m_max, r_mean, f_mean, m_mean, r_min, f_min, m_min,r,s):
    rs = init_clear_data(data=data,model=model,s=s)
    for index in rs.index:
        rs.loc[index, u'会员价值分数'] = comput(r_max,rs.loc[index]['R'],rs.loc[index]['F'],rs.loc[index]['M'])
        for i in range(8):
            if (rs.loc[index, u'聚类类别'] == r.index[i]):
                #rs.loc[index,u'个体聚类分数'] = r.loc[r.index[i],u'分数']
                r_score = comput_score(r_max, r_min, rs.loc[index]['R'], 1)
                f_score = comput_score(f_max, f_min, rs.loc[index]['F'], 0)
                m_score = comput_score(m_max, m_min, rs.loc[index]['M'], 0)
                score = comput(0,-r_score,f_score,m_score)
                if (score > r.loc[r.index[i],u'分数']):
                    rs.loc[index, u'波动'] = 1
                else:
                    rs.loc[index, u'波动'] = 2

        if rs.loc[index]['R'] < r_mean:
            if rs.loc[index]['F'] > f_mean:
                if rs.loc[index]['M'] > m_mean:
                    rs.loc[index, u'会员分类'] = H[0]
                else:
                    rs.loc[index, u'会员分类'] = H[1]
            else:
                if rs.loc[index]['M'] > m_mean:
                    rs.loc[index, u'会员分类'] = H[2]
                else:
                    rs.loc[index, u'会员分类'] = H[3]
        else:
            if rs.loc[index]['F'] > f_mean:
                if rs.loc[index]['M'] > m_mean:
                    rs.loc[index, u'会员分类'] = H[4]
                else:
                    rs.loc[index, u'会员分类'] = H[5]
            else:
                if rs.loc[index]['M'] > m_mean:
                    rs.loc[index, u'会员分类'] = H[6]
                else:
                    rs.loc[index, u'会员分类'] = H[7]
    return rs

def write_csv(rs_):
    rs_.to_csv(outputfile, encoding='utf-8')  # 保存结果

def main():
    data = read_data(input_file_=inputfile)
    data_zs, r_max, f_max, m_max, r_mean, f_mean, m_mean, r_min, f_min, m_min = clear_data(data_=data)
    model = k_means(data_zs=data_zs)
    r,s = clear_k_means(model=model)
    print(r)
    print(r_mean,f_mean,m_mean)
    rs = clear_s_data(data=data, model=model, r_max=r_max,f_max=f_max,m_max=m_max, r_mean=r_mean, f_mean=f_mean, m_mean=m_mean,r_min=r_min,f_min=f_min,m_min=m_min,r=r,s=s)
    write_csv(rs_=rs)
    print(rs)
if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('time',end-start)




