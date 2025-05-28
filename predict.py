from torch.library import define

from QM_BUPT_model import SquareRootFitter
mark = 75
x_data = [76, 86, 84, 59, 78, 80, 64]
y_data = [89, 94, 93, 77, 90, 91, 81]
# 有新的直接加到数组里
model = SquareRootFitter(x_data, y_data)
print("转换后的BUPT分数为:"+model.predict(mark))            # 单个预测

print(model.get_equation())  # 打印模型公式



model.show_fit()
