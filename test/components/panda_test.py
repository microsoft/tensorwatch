import pandas as pd

class SomeThing:
    def __init__(self, x, y):
        self.x, self.y = x, y

things = [SomeThing(1,2), SomeThing(3,4), SomeThing(4,5)]

df = pd.DataFrame([t.__dict__ for t in things ])

print(df.iloc[[-1]].to_dict('records')[0])
#print(df.to_html())
print(df.style.render())
