import numpy as np
import matplotlib.pyplot as plt

from buffer import TrendBuffer

buffer:TrendBuffer=TrendBuffer(10)

x=np.random.normal(0,1.0,10)

for _x in x:
    buffer.push(_x)

a=buffer.trendline()
print(a)

plt.plot(buffer.timeseries,x)

plt.plot(buffer.timeseries,buffer.timeseries*a)

plt.show()