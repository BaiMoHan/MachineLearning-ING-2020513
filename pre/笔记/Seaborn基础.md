# Seaborn基础

使用 Seaborn 完成图像快速优化的方法非常简单。只需要将 Seaborn 提供的样式声明代码 `sns.set()` 放置在绘图前即可。

`sns.set()` 的默认参数为：

```python
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
```

- `context=''` 参数控制着默认的画幅大小，分别有 `{paper, notebook, talk, poster}` 四个值。其中，`poster > talk > notebook > paper`。
- `style=''` 参数控制默认样式，分别有 `{darkgrid, whitegrid, dark, white, ticks}`，你可以自行更改查看它们之间的不同。
- `palette=''` 参数为预设的调色板。分别有 `{deep, muted, bright, pastel, dark, colorblind}` 等，你可以自行更改查看它们之间的不同。
- 剩下的 `font=''` 用于设置字体，`font_scale=` 设置字体大小，`color_codes=` 不使用调色板而采用先前的 `'r'` 等色彩缩写。