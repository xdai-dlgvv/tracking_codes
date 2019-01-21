# 数据规范化
def linear_mapping(img):
    max = img.max()
    min = img.min()

    parameter_a = 1 / (max - min)
    parameter_b = 1 - max * parameter_a

    img_after_linear_mapping = parameter_a * img + parameter_b

    return img_after_linear_mapping
