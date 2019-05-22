#import tensorwatch as tw
from tensorwatch import evaler


e = evaler.Evaler(expr='reduce(lambda x,y: (x+y), map(lambda x:(x**2), filter(lambda x: x%2==0, l)))')
for i in range(5):
    eval_return = e.post(i)
    print(i, eval_return)
eval_return = e.post(ended=True)
print(i, eval_return)
