import heapq
import numpy as np
path = 'recolab-data/collaborative/'



with open(path+'items.txt') as fil:
    items = fil.read()
    items = items.split('\n')
items = items[:-1]
items = np.array(items)
items = np.reshape(items,(items.shape[0],1))

M3=[]
with open(path+'ratings.txt') as fil:
    rate = fil.read().split('\n')
    for row in rate[:-1]:
        row = row.rstrip().split(" ")
        row = [int(r) for r in row]
        
        M3.append(row)

def user_cf(u1,u2):
    u = np.dot(u1,np.dot(u2,np.dot(u2.T,np.dot(u1,u2))))
    return u

def item_cf(i1,i2):
    i = np.dot(i2,np.dot(i1,np.dot(i2.T,np.dot(i2,i1))))
    return i        

rate = np.array(M3)
M1 = np.zeros((rate.shape[0],rate.shape[0]),dtype = 'int32')
M2 = np.zeros((rate.shape[1],rate.shape[1]),dtype = 'int32')
sum1 = np.reshape(np.sum(rate,axis = 1),(rate.shape[0],1))
sum2 = np.reshape(np.sum(rate,axis = 0),(rate.shape[1],1))


for i in range(rate.shape[0]):
    M1[i][i] = int(sum1[i][0])

for i in range(rate.shape[1]):
    M2[i][i] = int(sum2[i][0])

r1=M1.shape[0]
c1=M1.shape[1]
M1_inv = np.zeros((r1,c1))
r2=M2.shape[0]
c2=M2.shape[1]
M2_inv = np.zeros((r2,c2))
for i in range(r1):
    if M1[i,i] == 0:
        M1_inv[i,i] = 0
    else:
        M1_inv[i,i] = 1/M1[i,i]

for j in range(r2):
    if M2[j,j] == 0:
        M2_inv[j,j] = 0
    else:
        M2_inv[j,j] = 1/M2[j,j]

M1_inv = pow(M1_inv,0.5)
M2_inv = pow(M2_inv,0.5)

print('User recommendation')
user_r = user_cf(M1_inv,rate)

user_500 = user_r[499,:]

top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)



with open(path+'orig.txt') as f:
    orig = f.read().split(' ')
    orig = [int(i) for i in orig]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
print('Top 5 recommendations')
for i  in top_5:
    print(items[i])
print('Top 100 recommendations')
for i in top_100:
    print(items[i])

count_100 = 0
for i in top_100:
    if(orig[i] == 1):
        count_100 += 1

count_5 = 0
for i in top_5:
    if(orig[i] == 1):
        count_5 += 1
print('Number of correct predictions in top 100')
print(count_100)
print('Number of correct predictions in top 5')
print(count_5)

print('Item recommendation')
item_r = item_cf(M2_inv,rate)

user_500 = item_r[499,:]
top_100 = heapq.nlargest(100,range(len(user_500)),user_500.take)
top_5 = heapq.nlargest(5,range(len(user_500)),user_500.take)
print('Top 100 recommendations')
for i in top_100:
    print(items[i])
print('Top 5 recommendations')
for i in top_5:
    print(items[i])

count_100 = 0
count_5 = 0
for i in top_100:
    if(orig[i] == 1):
        count_100 += 1
for i in top_5:
    if(orig[i] == 1):
        count_5 += 1
print('Number of correct predictions in top 5')
print(count_5)
print('Number of correct predictions in top 100')
print(count_100)
