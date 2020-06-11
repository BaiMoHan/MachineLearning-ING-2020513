# Matplotlib入门





| 方法                               | 含义           |
| ---------------------------------- | -------------- |
| `matplotlib.pyplot.angle_spectrum` | 绘制电子波谱图 |
| `matplotlib.pyplot.bar`            | 绘制柱状图     |
| `matplotlib.pyplot.barh`           | 绘制直方图     |
| `matplotlib.pyplot.broken_barh`    | 绘制水平直方图 |
| `matplotlib.pyplot.contourf`       | 绘制等高线图   |
| `matplotlib.pyplot.errorbar`       | 绘制误差线     |
| `matplotlib.pyplot.hexbin`         | 绘制六边形图案 |
| `matplotlib.pyplot.hist`           | 绘制柱形图     |
| `matplotlib.pyplot.hist2d`         | 绘制水平柱状图 |
| `matplotlib.pyplot.pie`            | 绘制饼状图     |
| `matplotlib.pyplot.quiver`         | 绘制量场图     |
| `matplotlib.pyplot.scatter`        | 散点图         |
| `matplotlib.pyplot.specgram`       | 绘制光谱图     |





`matplotlib.pyplot.plot(*args, **kwargs)` 方法严格来讲可以绘制线形图或者样本标记。其中，`*args` 允许输入单个 $y$ 值或 $x, y$ 值。

线形图通过 `matplotlib.pyplot.plot(*args, **kwargs)` 方法绘出。其中，`args` 代表数据输入，而 `kwargs` 的部分就是用于设置样式参数了。

其中比较重要的样式参数为：

| 参数         | 含义                            |
| ------------ | ------------------------------- |
| `alpha=`     | 设置线型的透明度，从 0.0 到 1.0 |
| `color=`     | 设置线型的颜色                  |
| `fillstyle=` | 设置线型的填充样式              |
| `linestyle=` | 设置线型的样式                  |
| `linewidth=` | 设置线型的宽度                  |
| `marker=`    | 设置标记点的样式                |
| ……           | ……                              |



散点图主要参数：

| 参数          | 含义                 |
| ------------- | -------------------- |
| `s=`          | 散点大小             |
| `c=`          | 散点颜色             |
| `marker=`     | 散点样式             |
| `cmap=`       | 定义多类别散点的颜色 |
| `alpha=`      | 点的透明度           |
| `edgecolors=` | 散点边缘颜色         |

通过 [<i class="fa fa-external-link-square" aria-hidden="true"> 官方文档</i>](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) 详细了解。

任何图形的绘制，都建议通过 `plt.figure()` 或者 `plt.subplots()` 管理一个完整的图形对象。而不是简单使用一条语句，例如 `plt.plot(...)` 来绘图。

绘制包含图标题、坐标轴标题以及图例的图形，举例如下：

```python
fig, axes = plt.subplots()

axes.set_xlabel('x label')  # 横轴名称
axes.set_ylabel('y label')
axes.set_title('title')  # 图形名称

axes.plot(x, x**2)
axes.plot(x, x**3)
axes.legend(["y = x**2", "y = x**3"], loc=0)  # 图例1
```

图例中的 `loc` 参数标记图例位置，`1，2，3，4` 依次代表：右上角、左上角、左下角，右下角；`0` 代表自适应



对于线型而言，除了实线、虚线之外，还有很多丰富的线型可供选择。

```python
fig, ax = plt.subplots(figsize=(12, 6))

# 线宽
ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)

# 虚线类型
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# 虚线交错宽度
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10])

# 符号
ax.plot(x, x + 9, color="green", lw=2, ls='--', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')

# 符号大小和颜色
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-',
        marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8,
        markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")
```

