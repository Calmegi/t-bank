import numpy as np
import pandas as pd
from scipy import stats as sps
import math
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns
from statsmodels.stats.descriptivestats import sign_test

#1.1
#print("1.1")
data_frame = pd.read_excel("data_matstat_K5.xls", sheet_name = 'A - aaup')
A5, A7, A8, A4, A20 = np.genfromtxt("1.csv", delimiter=';')[1:].transpose()

#for i in [A5, A7, A8] :
    #print(print(sps.describe(i)))
  
#1.2
#print("1.2")

n = len(A7)
#print(f"n = {n}")

#формула Стеджерса

k = int(1 + math.log2(n))
#print(f"k = {k}")

step = int((A7.max() - A7.min())/k)
#print(f"step = {step}")

frequencies, edges, _ = plt.hist(A7, bins = k)
midpoints = 0.5 * (edges[1:] + edges[:-1])
print(frequencies)
print(edges[:-1])
print(edges[1:])
plt.plot(midpoints, frequencies)

relative_frequencies, edges, _ = plt.hist(A7, bins = k, weights = np.zeros_like(A7) + 1/n)
midpoints = 0.5 * (edges[1:] + edges[:-1])
print(relative_frequencies)
plt.plot(midpoints, relative_frequencies)

accumulated_frequencies, edges, _ = plt.hist(A7, bins = k, cumulative=True)
midpoints = 0.5 * (edges[1:] + edges[:-1])
print(accumulated_frequencies)
plt.plot(midpoints, accumulated_frequencies)

accumulated_relative_frequencies, edges, _ = plt.hist(A7, bins = k, weights = np.zeros_like(A7) + 1/n, cumulative=True)
midpoints = 0.5 * (edges[1:] + edges[:-1])
print(accumulated_relative_frequencies)
plt.plot(midpoints, accumulated_relative_frequencies)

ecdf = ECDF(A7)
plt.plot(ecdf.x, ecdf.y, lw=3)
plt.xlabel("$x$")
plt.ylabel("$F(x)$")

#2

alphas = [0.01, 0.05, 0.1]

#2.1
for alpha in alphas:
    interval = (A7.std() / n ** 0.5) * sps.t.ppf(1 - alpha / 2, n - 1)
    print(alpha)
    print("lower edge:  ", A7.mean() - interval)
    print("higher edge:  ", A7.mean() + interval)

#2.2
for alpha in alphas:
    print(alpha)
    lower_edge = (n - 1) * A7.var() / (sps.chi2.ppf(1 - alpha / 2, n - 1))
    higher_edge = (n - 1) * A7.var() / (sps.chi2.ppf(alpha / 2, n - 1))
    print("lower edge:  ", lower_edge)
    print("higher edge:  ", higher_edge)


#2.3
n1 = len(A7)
s1 = A7.std()
n2 = len(A8)
s2 = A8.std()
s = (((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) ** 0.5

for alpha in alphas:
    interval = sps.t.ppf(1 - alpha/2, n1 + n2 - 2) * s * (1/n1 + 1/n2) ** 0.5
    print(alpha)
    print("lower edge:  ", A7.mean() - A8.mean() - interval)
    print("higher edge:  ", A7.mean() - A8.mean() + interval)


#2.4
for alpha in alphas:
    print(alpha)
    lower_edge = (A7.var() / A8.var()) * (sps.f.ppf(alpha/2, n2 - 1, n1 - 1))
    higher_edge = (A7.var() / A8.var()) * (sps.f.ppf(1 - alpha/2, n2 - 1, n1 - 1))
    print("lower edge:  ", lower_edge)
    print("higher edge:  ", higher_edge)



#3.1

print(A7.mean())
print(A7.std())

print(sps.ttest_1samp(A7, 390))

#3.2


def one_sample_chi_test_unknown(x, sigma0):
    S_squared = 1 / (len(x) - 1) * sum([(i - x.mean()) ** 2 for i in x])
    z = (len(x) - 1) * S_squared / sigma0 ** 2
    f = sps.chi2.cdf(z, df=len(x)-1)
    return z, 2 * min(f, 1 - f)


print(one_sample_chi_test_unknown(A7, 70))


#3.3

print(A7.mean(), A8.mean())

def two_sample_t_test(x, y):
    S_1 = (1 / (len(x) - 1) * sum([(i - x.mean()) ** 2 for i in x])) 
    S_2 = (1 / (len(y) - 1) * sum([(i - y.mean()) ** 2 for i in y]))
    S = ((len(x) - 1) * S_1 + (len(y) - 1) * S_2) / (len(x) + len(y) - 2)
    z = (x.mean() - y.mean()) * (1 / len(x) + 1 / len(y)) ** 0.5 / S ** 0.5
    f = sps.t.cdf(z, df=len(x)+len(y)-2)
    return z, 2 * min(f, 1 - f)

print(two_sample_t_test(A7, A8))


#sigma1 != sigma2
def two_sample_t_test_2(x, y):
    S_1 = (1 / (len(x) - 1) * sum([(i - x.mean()) ** 2 for i in x])) 
    S_2 = (1 / (len(y) - 1) * sum([(i - y.mean()) ** 2 for i in y]))
    z = (x.mean() - y.mean()) / (S_1 / len(x) + S_2 / len(y)) ** 0.5
    t1 = S_1  / len(x)
    t2 = S_2 / len(y)
    k = (t1 / (t1 + t2)) ** 2 / (len(x) - 1) + (t2 / (t1 + t2)) ** 2 / (len(y) - 1)
    f = sps.t.cdf(z, df=int(1 / k))
    return z, 2 * min(f, 1 - f)

print(two_sample_t_test_2(A7, A8))


#3.4

def two_sample_f_test(x, y):
    S_1 = 1 / (len(x) - 1) * sum([(i - x.mean()) ** 2 for i in x])
    S_2 = 1 / (len(y) - 1) * sum([(i - y.mean()) ** 2 for i in y])
    z = S_1 / S_2
    f = sps.f.cdf(z, dfn=len(x)-1, dfd=len(y)-1)
    return z, 2 * min(f, 1 - f)


print(two_sample_f_test(A7, A8))




#4.1

#формула Стеджерса

k = int(1 + math.log2(n))
#print(f"k = {k}")

step = int((A7.max() - A7.min())/k)
#print(f"step = {step}")

frequencies, edges, _ = plt.hist(A7, bins = k)
midpoints = 0.5 * (edges[1:] + edges[:-1])
#print(frequencies)
#print(edges[:-1])
#print(edges[1:])
plt.plot(midpoints, frequencies)

relative_frequencies, edges, _ = plt.hist(A7, bins = k, weights = np.zeros_like(A7) + 1/n)
midpoints = 0.5 * (edges[1:] + edges[:-1])
#print(relative_frequencies)
plt.plot(midpoints, relative_frequencies)


loc = A7.mean()
scale = A7.std()
#print(loc, scale)
probabilities = [sps.norm.cdf(edges[i], loc=loc, scale=scale) - sps.norm.cdf(edges[i - 1], loc=loc, scale=scale) for i in range(1, k + 1) ]
#print(probabilities)

sns_plot = sns.distplot(A7, bins=k)
fig = sns_plot.get_figure()



f_1 = np.array([3.,35.,132.,235.,279.,199.,126.,35.,19.,8.,2.])
f_2 = np.array([n * probabilities[i] for i in range(k)])

m1 = sps.describe(f_1)[2] #индекс [2] обозначает среднее значение
m2 = sps.describe(f_2)[2]

print(sps.chisquare(f_obs=f_1, f_exp = f_2 * float(m1 / m2), ddof=2))

print(sps.jarque_bera(A7))

#5
#5.1

#print(sign_test(A7, A8))

#5.2

left = min(min(A7), min(A8))
right = max(max(A7), max(A8))
step = (right - left) / k
bins = [left + step * i for i in range(0, k + 1)]

frequencies_1, edges_1, _ = plt.hist(A7, bins=bins, alpha=0.5)
midpoints_1 = 0.5 * (edges_1[1:] + edges_1[:-1])
#print(frequencies_1)
#print(edges_1[:-1])
#print(edges_1[1:])

frequencies_2, edges_2, _ = plt.hist(A8, bins=bins, alpha=0.5)
midpoints_2 = 0.5 * (edges_2[1:] + edges_2[:-1])
#print(frequencies_2)
#print(edges_2[:-1])
#print(edges_2[1:])


relative_frequencies_1, edges_1, _ = plt.hist(A7, bins = bins, weights = np.zeros_like(A7) + 1/n)
midpoints_1 = 0.5 * (edges_1[1:] + edges_1[:-1])
print(relative_frequencies_1)
print()


relative_frequencies_2, edges_2, _ = plt.hist(A8, bins=bins, alpha=0.5, weights=np.zeros_like(A8) + 1 / n)
midpoints_2 = 0.5 * (edges_2[1:] + edges_2[:-1])
print(relative_frequencies_2)

plt.hist(A7, bins = bins, weights = np.zeros_like(A7) + 1/n)
plt.hist(A8, bins=bins, alpha=0.5, weights=np.zeros_like(A8) + 1 / n)
plt.legend()
print(sps.chisquare(f_obs=relative_frequencies_1, f_exp=relative_frequencies_2, ddof=2))



#6


n = len(A4)
print(n)


emp_table = pd.crosstab(index=data_frame["A4"], columns=data_frame["A20"], margins=True)
print(emp_table)


teor_table = sps.chi2_contingency(emp_table)[3]
print(teor_table)

print(sps.chi2_contingency(emp_table)[:2])



#7

#print(data_frame["A4"].unique())

f_c = data_frame.groupby(["A4"])["A4"].count()
#print(f_c)


n1 = f_c[0]
n2 = f_c[1]
n3 = f_c[2]
n = n1 + n2 + n3

i = data_frame[data_frame["A4"] == "I"]
iia = data_frame[data_frame["A4"] == "IIA"]
iib = data_frame[data_frame["A4"] == "IIB"]

#print(iib["A7"].mean())
#print(iib["A7"].var())

mean = (n1 * i["A7"].mean() + n2 * iia["A7"].mean() + n3 * iib["A7"].mean()) / n
k = 3

D_b = (n1 * (i["A7"].mean() - mean) ** 2 + n2 * (iia["A7"].mean() - mean) ** 2 + n3 * (iib["A7"].mean() - mean) ** 2) / n
#print(D_b)

D_w = (n1 * i["A7"].var() + n2 * iia["A7"].var() + n3 * iib["A7"].var()) / n
#print(D_w)

X = [i["A7"], iia["A7"], iib["A7"]]
sum = 0
for x in X:
  for i in x:
    sum += (i - mean) ** 2
D_x = sum / n
#print(D_x)


#print(n/(k-1)*D_b, n/(n-k)*D_w, n/(n-1)*D_x)


#print(D_b+D_w)
#print(D_b/D_w, math.sqrt(D_b/D_w))
X = pd.read_excel(open('data_matstat_K5.xls', 'rb'), sheet_name='A - aaup')['A4'].to_numpy()
Y = pd.read_excel(open('data_matstat_K5.xls', 'rb'), sheet_name='A - aaup')['A7'].to_numpy()

data = {'I': [], 'IIA': [], 'IIB': []}
for i in range(len(X)):
    if X[i] == 'I':
        data["I"].append(Y[i])
    elif X[i] == 'IIA':
        data["IIA"].append(Y[i])
    elif X[i] == 'IIB':
        data["IIB"].append(Y[i])
        
groups = []
for key in data:
    arr = np.array(data[key])
    groups.append(arr)
    print(f"Объем выборки {key}:",  len(data[key]))
    print(f"Cреднее {key}: {round(arr.mean(), 2)}")
    print(f"Дисперсия {key}: {round(arr.var(ddof=1), 2)}")
    
    
print(sps.f_oneway(*groups))
#sps.f_oneway(i["A7"], iia["A7"], iib["A7"])





#8.1

r, p_value_r = sps.pearsonr(data_frame["A7"], data_frame["A8"])
print(r, p_value_r, sep="\n")

rsp, p_value_rsp = sps.spearmanr(data_frame["A7"], data_frame["A8"])
print(rsp, p_value_rsp, sep="\n")

tau, p_value_tau = sps.kendalltau(data_frame["A7"], data_frame["A8"])
print(rsp, p_value_rsp, sep="\n")

alpha = [0.01, 0.05, 0.1]
a = r + (r * (1 - r ** 2)) / (2 * n)
for a in alpha:
  b = sps.norm.ppf(1 - a / 2, (1 - r ** 2) / math.sqrt(n))
  print("alpha=", a, ":", sep="")
  print("lower edge:", a - b)
  print("higher edge:", a + b)
  print()

z_r = (r / math.sqrt(1 - r ** 2)) * math.sqrt(n - 2)
print(z_r)

z_rsp = (rsp / math.sqrt(1 - rsp ** 2)) * math.sqrt(n - 2)
print(z_rsp)

z_r_tau = tau * math.sqrt(9 * n * (n + 1) / (2 * (2 * n + 5)))
print(z_r_tau)


#8.2

data = {"A5": data_frame["A5"], "A7": data_frame["A7"], "A8": data_frame["A8"]}

df = pd.DataFrame(data)
#print(df.corr(method="kendall"))


data_p = {
    "A5": [sps.kendalltau(data_frame["A5"], data_frame["A5"])[1], sps.kendalltau(data_frame["A5"], data_frame["A7"])[1], sps.kendalltau(data_frame["A5"], data_frame["A8"])[1]], 
    "A7": [sps.kendalltau(data_frame["A7"], data_frame["A5"])[1], sps.kendalltau(data_frame["A7"], data_frame["A7"])[1], sps.kendalltau(data_frame["A7"], data_frame["A8"])[1]],
    "A8": [sps.kendalltau(data_frame["A8"], data_frame["A5"])[1], sps.kendalltau(data_frame["A8"], data_frame["A7"])[1], sps.kendalltau(data_frame["A8"], data_frame["A8"])[1]]
    }
df_p = pd.DataFrame(data_p)
#print(df_p)


R1 = sps.rankdata(data_frame["A5"])
R2 = sps.rankdata(data_frame["A7"])
R3 = sps.rankdata(data_frame["A8"])

k = 3 #число выборок
sum1 = 0
sum2 = 0
n = 1073
for i in range(n):
    for r in [R1, R2, R3]:
        sum1 += r[i]
    sum2 += (sum1 - k * (n + 1) / 2) ** 2
    sum1 = 0
W = 12 * sum2 / ((n ** 3 - n) * (k ** 2))
#print(W)

Z = n * (k - 1) * W
print(Z)

a = 0.1
kr_p = sps.chi2.ppf(a, n - 1)
print(kr_p)

#Получила, что значение Z больше, чем значение критической точки, то есть коэффициент конкордации значим

print(1 - sps.chi2(n-1).cdf(Z))

p_value = 1 - sps.chi2.cdf(Z, n - 1)
print(p_value)


#9
n = 1073
k = 2

x = data_frame["A17"]
y = data_frame["A12"]

r, p_v = sps.pearsonr(x, y)

b0 = y.mean() - r * x.mean() * y.std() / x.std()
b1 = r * y.std() / x.std()


sum = 0
for elem in x:
    sum += (b0 + b1 * elem - y.mean()) ** 2
D_r = sum / n

sum = 0
a = list(zip(x, y))
for elem in a:
  sum += (elem[1] - (b0 + b1 * elem[0])) ** 2
D_ost = sum / n

sum = 0
for elem in y:
  sum += (elem - y.mean()) ** 2
D_y = sum / n


est_r = n * D_r / (k - 1)
est_ost = n * D_ost / (n - k)
est_y = n * D_y / (n - 1)
#print(est_r, est_ost, est_y, sep="\n")

#print(D_r + D_ost)


R_sq = D_r / D_y
R = math.sqrt(D_r / D_y)
#print(R_sq, R, sep='\n')

alpha = [0.01, 0.05, 0.1]

print("b0")
for a in alpha:
    z = sps.t.ppf(1 - a / 2, n - 2) * math.sqrt(D_ost) * math.sqrt(np.sum([elem ** 2 for elem in x]) / (n ** 2 * x.var()))
    print(a)
    print("l:", b0 - z)
    print("h:", b0 + z)
    print()

print("b1")
for a in alpha:
      z = sps.t.ppf(1 - a / 2, n - 2) * math.sqrt(D_ost / (n * x.var())) 
      print(a)
      print("l:", b1 - z)
      print("h:", b1 + z)
      print()

#sns.regplot(x, y)
plt.plot(x, [elem_y - (b0 + b1 * elem_x) for elem_x, elem_y in zip(x, y)], "o")
plt.show()


statistics = R_sq * (n - 2) / (1 - R_sq)
p_value = 1 - sps.f.cdf(statistics, 1, n - 2)
print(statistics, p_value)


#9.2
#9.2.1

n = 1073
k = 3

x = data_frame["A17"]
y = data_frame["A12"]

F = np.array([[1, elem, elem ** 2] for elem in x])
b = np.linalg.inv((F.T).dot(F)).dot(F.T).dot(np.array(y))

b0 = b[0]
b1 = b[1]
b2 = b[2]

sum = 0
for elem in x:
  sum += (b0 + b1 * elem + b2 * elem ** 2  - y.mean()) ** 2
D_r = sum / n

sum = 0
a = list(zip(x, y))
for elem in a:
  sum += (elem[1] - (b0 + b1 * elem[0] + b2 * elem[0] ** 2)) ** 2
D_ost = sum / n

sum = 0
for elem in y:
  sum += (elem - y.mean()) ** 2
D_y = sum / n

k = 3
est_r = n * D_r / (k - 1)
est_ost = n * D_ost / (n - k)
est_y = n * D_y / (n - 1)


D_y_sum = D_r + D_ost

R_sq = D_r / D_y
R = math.sqrt(D_r / D_y)

#9.2.2


b = sps.t.ppf(1 - 0.1 / 2, n - 3) * math.sqrt(D_ost)
c = np.linalg.inv((F.T).dot(F))
x_s = sorted(list(x))
#plt.plot(x, y, 'o')
#plt.plot(x_s, [b0 + b1 * elem + b2 * elem ** 2 for elem in x_s], 'r')
#plt.show()



#plt.plot(x, [elem_y - (b0 + b1 * elem_x + b2 * elem_x ** 2) for elem_x, elem_y in zip(x, y)], "o")
#plt.show()



k = 3
statistics = R_sq * (n - k) / ((1 - R_sq) * (k - 1))
p_value = 1 - sps.f.cdf(statistics, k - 1, n - k)
print(statistics, p_value)



#9.3
#9.3.1

n = 1073
k = 3

x1 = data_frame["A17"]
x2 = data_frame["A7"]
y = data_frame["A12"]

F = np.array([[1, elem_x1, elem_x2] for elem_x1, elem_x2 in zip(x1, x2)])
b = np.linalg.inv((F.T).dot(F)).dot(F.T).dot(np.array(y))

b0 = b[0]
b1 = b[1]
b2 = b[2]

D_r = np.sum([(b0 + b1 * elem_x1 + b2 * elem_x2 - y.mean()) ** 2 for elem_x1, elem_x2 in zip(x1, x2)]) / n
print(D_r)
D_ost = np.sum([(elem_y - (b0 + b1 * elem_x1 + b2 * elem_x2)) ** 2 for elem_x1, elem_x2, elem_y in zip(x1, x2, y)]) / n
print(D_ost)
D_y = np.sum([(elem_y - y.mean()) ** 2 for elem_y in y]) / n
print(D_y)
k = 3
est_r = n * D_r / (k - 1)
est_ost = n * D_ost / (n - k)
est_y = n * D_y / (n - 1)
#print(est_r, est_ost, est_y, sep="\n")

D_y_sum = D_r + D_ost
print(D_y_sum)


R_sq = D_r / D_y
R = math.sqrt(D_r / D_y)
print(R_sq, R, sep='\n')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y,  color='black')
ax.set_xlabel('X1 Label')
ax.set_ylabel('X2 Label')
ax.set_zlabel('Y Label')

plt.show()

statistics = R_sq * (n - k) / ((1 - R_sq) * (k - 1))
p_value = 1 - sps.f.cdf(statistics, k - 1, n - k)
print(statistics, p_value)

n = 1073
k = 3

x = data_frame["A17"]
y = data_frame["A12"]

F = np.array([np.array([1]*len(x)), x, x ** 2]).transpose()
dF = np.dot(F.transpose(), F)
result = np.dot(np.linalg.inv(dF), F.transpose()).dot(y)
beta_0 = result[0]
beta_1 = result[1]
beta_2 = result[2]
print(f"beta0 = {result[0]} \nbeta1 = {result[1]} \nbeta2 = {result[2]}")

def f(x):
    return beta_0 + beta_1 * x + beta_2 * x ** 2


data = {
    'X': x,
    'Y': f(x)
}
df1 = pd.DataFrame(data)

X_grp = df1.groupby('X')


n = len(y)
D_YX = np.sum(X_grp.count().to_numpy() * (X_grp.mean().to_numpy() - np.mean(y)) ** 2) * 1 / n

D_resY = np.sum((y - f(x)) ** 2) / n


D_res = np.var(y)


print(f"{D_YX = }")
print(f"{D_resY = }")
print(f"{D_res = }")
print(f"{D_YX + D_resY = }")
est_r = n * D_YX / (k - 1)
est_ost = n * D_resY / (n - k)
est_y = n * D_res / (n - 1)
print(est_r, est_ost, est_y, sep="\n")
RXY = D_YX/D_res
print (RXY)
print(math.sqrt(RXY))

X1 = np.sort(x)
new_D_res = np.sum((y - f(x)) ** 2) / (n - 2)
quantile = sps.t(n - 3).ppf(1- 0.1/2)

x_arr = np.array([1, X1, X1 ** 2])
flow = f(X1) - quantile * np.sqrt(new_D_res * x_arr.transpose() @ np.linalg.inv(dF) @ x_arr)
fhigh = f(X1) + quantile * np.sqrt(new_D_res * x_arr.transpose() @ np.linalg.inv(dF) @ x_arr)

b = sps.t.ppf(1 - 0.1 / 2, n - 3) * math.sqrt(D_resY)
c = np.linalg.inv((F.T).dot(F))
x_s = sorted(list(x))




plt.plot(x, [elem_y - (beta_0 + beta_1 * elem_x + beta_2 * elem_x ** 2) for elem_x, elem_y in zip(x, y)], "o")
plt.show()


R_sq = D_YX / D_res
k = 3
statistics = R_sq * (n - k) / ((1 - R_sq) * (k - 1))
p_value = 1 - sps.f.cdf(statistics, k - 1, n - k)
print(statistics, p_value)

