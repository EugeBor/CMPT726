x = [-5,-4,-1,4,5,6,7,8,9,10]

i=0
g=x
while i < 10:
    g[i] = max(-1+x[i],0)
    print(g[i])
    i += 1