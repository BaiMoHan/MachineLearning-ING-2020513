# Numpy数值计算基础入门

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

##1. NumPy中的数据类型
|    类型    |                          解释                          |
| :--------: | :----------------------------------------------------: |
|    bool    |        布尔类型，1 个字节，值为 True 或 False。        |
|    int     |           整数类型，通常为 int64 或 int32 。           |
|    intc    |      与 C 里的 int 相同，通常为 int32 或 int64。       |
|    intp    |           用于索引，通常为 int32 或 int64。            |
|    int8    |                 字节（从 -128 到 127）                 |
|   int16    |               整数（从 -32768 到 32767）               |
|   int32    |          整数（从 -2147483648 到 2147483647）          |
|   int64    | 整数（从 -9223372036854775808 到 9223372036854775807） |
|   uint8    |               无符号整数（从 0 到 255）                |
|   uint16   |              无符号整数（从 0 到 65535）               |
|   uint32   |            无符号整数（从 0 到 4294967295）            |
|   uint64   |       无符号整数（从 0 到 18446744073709551615）       |
|   float    |                    float64 的简写。                    |
|  float16   |            半精度浮点，5 位指数，10 位尾数             |
|  float32   |            单精度浮点，8 位指数，23 位尾数             |
|  float64   |            双精度浮点，11 位指数，52 位尾数            |
|  complex   |                  complex128 的简写。                   |
| complex64  |              复数，由两个 32 位浮点表示。              |
| complex128 |              复数，由两个 64 位浮点表示。              |


在 NumPy 中，上面提到的这些数值类型都被归于 `dtype（data-type）` 对象的实例。
我们可以用 `numpy.dtype(object, align, copy)` 来指定数值类型。而在数组里面，可以用 `dtype=` 参数。

在Jupyter Notebook中运行以下代码，后续代码均在此环境下
```python
import numpy as np # 导入NumPy模块

a = np.array([1.1,2.2,3.3],dtype=np.float64) # 指定1维度数组的类型为float64
a, a.dtype # 查看a及dtype类型
```
一般在导入的时候都另名为np，方便后面代码输入
```python
a.astype(int).dtype #将a的数值类型从float64转换为int，并查看dtype类型
```

##2. NumPy中的数组

Python中内置的三种形式的数组：
- 列表：`[1, 2, 3]`
- 元组：`(1, 2, 3, 4, 5)`
- 字典：`{A:1, B:2}`
> NumPy 最核心且最重要的一个特性就是 ndarray 多维数组对象，它区别于 Python 的标准类，拥有对高维数组的处理能力，这也是数值计算过程中缺一不可的重要特性。

NumPy 中，`ndarray` 类具有六个参数，它们分别为：
- `shape`：数组的形状。
- `dtype`：数据类型。
- `buffer`：对象暴露缓冲区接口。
- `offset`：数组数据的偏移量。
- `strides`：数据步长。
- `order`：`{'C'，'F'}`，以行或列为主排列顺序。

## 3.NumPy中创建数组

在 NumPy 中，我们主要通过以下 5 种途径创建数组:
- 从 Python 数组结构列表，元组等转换。
- 使用 `np.arange`、`np.ones`、`np.zeros` 等 NumPy 原生方法。
- 从存储空间读取数组。
- 通过使用字符串或缓冲区从原始字节创建数组。
- 使用特殊函数，如 `random`。
###3.1 列表或元组的转换
使用 `numpy.array` 将列表或元组转换为 `ndarray` 数组。其方法为：
```python
numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0)
```
参数如下：
 - `object`：列表、元组等。
 - `dtype`：数据类型。如果未给出，则类型为被保存对象所需的最小类型。
 - `copy`：布尔类型，默认 True，表示复制对象。
 - `order`：顺序。
 - `subok`：布尔类型，表示子类是否被传递。
 - `ndmin`：生成的数组应具有的最小维数。

通过列表创建一个`ndarray`数组
```python
np.array([[1, 2, 3], [4, 5, 6]])
```
通过元组创建一个`ndarray`数组
```python
np.ndarray([(1,2),(11,22),(111,222)])
```
###3.2 NumPy原生方法创建`ndarray`
####3.2.1 arange方法创建
```python
numpy.arange(start, stop, step, dtype=None)
```
先设置值所在的区间 `[开始， 停止)`，这是一个半开半闭区间。然后，在设置 `step` 步长用于设置值之间的间隔。最后的可选参数 `dtype `可以设置返回`ndarray` 的值类型。
```python
np.arange(3,7,0.5,dtype='int32')
```
####3.2.2 linspace方法创建
```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
```
- `start`：序列的起始值。
- `stop`：序列的结束值。
- `num`：生成的样本数。默认值为50。
- `endpoint`：布尔值，如果为真，则最后一个样本包含在序列内。
- `retstep`：布尔值，如果为真，返回间距。
- `dtype`：数组的类型。
创建数值有规律的数组。`linspace` 用于在指定的区间内返回间隔均匀的值

```python
np.linspace(0,20,15,endpoint=True)
```
####3.2.3 ones方法创建
`numpy.ones` 用于快速创建数值全部为 `1` 的多维数组。其方法如下：
```python
numpy.ones(shape, dtype=None, order='C')
```
- `shape`：用于指定数组形状，例如（1， 2）或 3。
- `dtype`：数据类型。
- `order`：`{'C'，'F'}`，按行或列方式储存数组。

```python
np.ones((2,5),order='C')
```
####3.2.4 zeros方法创建
`zeros` 方法和上面的 `ones` 方法非常相似，不同的地方在于，这里全部填充为 `0`
```python
numpy.zeros(shape, dtype=None, order='C')
```
- `shape`：用于指定数组形状，例如`（1， 2）`或` 3`。
- `dtype`：数据类型。
- `order`：`{'C'，'F'}`，按行或列方式储存数组

```python
np.zeros((3,5))
```

####3.2.5 eye方法创建

`numpy.eye` 用于创建一个二维数组，其特点是` k` 对角线上的值为 `1`，其余值全部为` 0`。方法如下：
```python 
numpy.eye(N, M=None, k=0, dtype=<type 'float'>)`
```

- `N`：输出数组的行数。
- `M`：输出数组的列数。
- `k`：对角线索引：0（默认）是指主对角线，其他数值可以理解为从主对角线向外数。正值是指上对角线，负值是指下对角线。
```python
np.eye(5, 4, -2)
```
###3.3 从已知数据创建
从已知数据文件、函数中创建 `ndarray`。NumPy 提供了下面 `5` 个方法：
- `frombuffer（buffer）`：将缓冲区转换为 `1` 维数组。
- `fromfile（file，dtype，count，sep）`：从文本或二进制文件中构建多维数组。
- `fromfunction（function，shape）`：通过函数返回值来创建多维数组。
- `fromiter（iterable，dtype，count）`：从可迭代对象创建 `1` 维数组。
- `fromstring（string，dtype，count，sep）`：从字符串中创建 `1` 维数组。
通过`lambda`表达式返回值创建多维数组
```python
np.fromfunction(lambda a, b: a + b, (7, 8))
```

##4 `ndarray`数组
###4.1 `ndarray`数组属性
`ndarray.T` 用于数组的转置，与 `.transpose()` 相同。
`ndarray.dtype` 用来输出数组包含元素的数据类型。
`ndarray.imag` 用来输出数组包含元素的虚部。
`ndarray.real`用来输出数组包含元素的实部。
`ndarray.size`用来输出数组中的总包含元素数。
`ndarray.itemsize`输出一个数组元素的字节数。
`ndarray.nbytes`用来输出数组的元素总字节数。
`ndarray.ndim`用来输出数组维度。
`ndarray.shape`用来输出数组形状。
`ndarray.strides`用来遍历数组时，输出每个维度中步进的字节数组。 `ndarray.shape` 属性查看 NumPy 数组的形状。

###4.2 `ndarray`基本操作

