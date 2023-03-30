# Провести дисперсионный анализ для определения того, есть ли различия среднего роста среди взрослых футболистов, хоккеистов и штангистов.
# Даны значения роста в трех группах случайно выбранных спортсменов:
# Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
# Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
# Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.

import numpy as np
from scipy import stats
from scipy.stats import bartlett
from scipy.stats import shapiro
import pandas as pd
from scipy. stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

a=np.array([173, 175, 180, 178, 177, 185, 183, 182])
b=np.array([177, 179, 180, 188, 177, 172, 171, 184, 180])
c=np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170])
print(stats.shapiro(a))                                             # ShapiroResult(statistic=0.9775082468986511, pvalue=0.9495404362678528)
print(stats.shapiro(b))                                             # ShapiroResult(statistic=0.9579196572303772, pvalue=0.7763139009475708)
print(stats.shapiro(c))                                             # ShapiroResult(statistic=0.9386808276176453, pvalue=0.5051165223121643)
# по результатам теста Шапиро: во всех случаях pvalue > 0.05, что означает, что все выборки имеют нормальное распределение

print(stats.bartlett(a,b,c))                                        # BartlettResult(statistic=0.4640521043406442, pvalue=0.7929254656083131)
# по результатам теста на однородность дисперсии pvalue > 0.05, что означает, что выборки однородны, т.е. равенство дисперсий

# Переходим к дисперсионному анализу, т.к. условия его применимости соблюдены:

a_mean=np.mean(a)
print (a_mean)                                                      # 179.125 - средний рост среди футболистов
b_mean=np.mean(b)
print (b_mean)                                                      # 178.66666666666666 - средний рост среди хоккеистов
c_mean=np.mean(c)
print (c_mean)                                                      # 172.72727272727272 - средний рост среди штангистов

k=3                                                                 # кол-во выборок

total=np.hstack([a,b,c])
print(total)
n=len(total)                                                        # 28 - число элементов во всех выборках
print(n)                                                          
total_mean=np.mean(total)
print(total_mean)                                                   # 176.46428571428572 - средний рост среди всех спортсменов 

# сумма квадратов отклонений наблюдений от общего среднего
print(np.sum((total-total_mean)**2))                                # 830.9642857142854

# сумма квадратов отклонений средних групповых значений от общего среднего
S_f2=np.sum((a_mean-total_mean)**2)*len(a)+np.sum((b_mean-total_mean)**2)*len(b)+np.sum((c_mean-total_mean)**2)*len(c)   # 253.9074675324678
print(S_f2)

# остаточная сумма квадратов отклонений
S_ost2=np.sum((a-a_mean)**2)+np.sum((b-b_mean)**2)+np.sum((c-c_mean)**2)                                                 # 577.0568181818182                      
print(S_ost2)

print(S_f2+S_ost2)                            # 830.964285714286, что равно сумме квадратов отклонений наблюдений от общего среднего (см.выше)

D_f=S_f2/(k-1)                                # 126.9537337662339
print(D_f)
D_ost=S_ost2/(n-k)                            # 23.08227272727273
print(D_ost)                                          

F_n=D_f/D_ost                                 # Наблюдаемый критерий Фишера F_n=5.500053450812598
print(F_n)

f=stats.f_oneway(a,b,c)                       # Определение наблюдаемого критерия Фишера с помощью функции
print(f)                                      # F_onewayResult(statistic=5.500053450812596, pvalue=0.010482206918698693)

F_t=3.38                                      # Табличное значение критерия Фишера (при k1=k-1=3-1=2 и k2=n-k=28-3=25)
F_n > F_t                                     # то гипотеза H1, т.е. что есть различия среднего роста среди футболистов, хоккеистов и штангистов

# Тест Тьюки  ??????? Как сделать для  несбалансированных данных?
df=pd.DataFrame({'score':list(total),'group':np.repeat(['a','b','c'],repeats=28)})
tukey=pairwise_tukeyhsd(endog=df['score'],groups=df['group'],alpha=0.05)
print(tukey)