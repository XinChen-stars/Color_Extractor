from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import colorsys
import yaml
import os


def extract_color_palette(image_path, num_colors=5, resize_factor=0.5, rgb_threshold=40):
    # 打开图像并转换为RGB
    image = Image.open(image_path)
    image = image.convert('RGB')

    # 缩小图像尺寸
    original_size = image.size
    new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
    image = image.resize(new_size, Image.LANCZOS)
    
    # 将图像数据转换为numpy数组
    image_np = np.array(image)

    # 将图像数据重塑为二维数组
    pixels = image_np.reshape(-1, 3)

    # 使用KMeans聚类来提取主要颜色
    kmeans = KMeans(n_clusters=3 * num_colors, random_state=42)
    kmeans.fit(pixels)

    # 获取聚类中心（即主要颜色）
    colors = kmeans.cluster_centers_.astype(int)

    # 过滤掉接近黑色和白色的颜色
    filtered_colors = []
    for color in colors:
        # 确保每个RGB分量都高于阈值（过滤掉接近黑色）并且低于255减去阈值（过滤掉接近白色）
        if all(c > rgb_threshold for c in color) and all(c < 255 - rgb_threshold for c in color):
            filtered_colors.append(color)
    
    # 如果过滤后颜色不足，补充原始颜色
    if len(filtered_colors) < num_colors:
        filtered_colors += list(colors[:num_colors - len(filtered_colors)])
    
    return np.array(filtered_colors[:num_colors])

def sort_colors_by_hue(colors):
    # 将RGB颜色转换为HSL并按色相进行排序
    def rgb_to_hsl(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        return colorsys.rgb_to_hls(r, g, b)
    
    colors_hsl = [rgb_to_hsl(color) for color in colors]
    sorted_colors_hsl = sorted(colors_hsl, key=lambda x: (x[0], x[2], x[1]))  # 按色相、亮度、饱和度排序
    sorted_colors_rgb = [tuple(int(c * 255) for c in colorsys.hls_to_rgb(h, l, s)) for h, l, s in sorted_colors_hsl]
    return sorted_colors_rgb

def plot_image_and_palette(image_path, colors):
     # 打开图像
    image = Image.open(image_path)

    # 缩小图像尺寸
    original_size = image.size
    new_size = (int(original_size[0] * resize_factor), int(original_size[1] * resize_factor))
    image = image.resize(new_size, Image.LANCZOS)

    # 计算合适的figsize
    # 图像宽度根据缩放后的宽度调整，调节比例因子以适应图形显示
    # fig_width = max(new_size[0] / 100, len(colors) * 1.2)  # 调整比例因子
    # fig_height = new_size[1] / 100 + 1 # 适当增加高度以给文本留出空间
    fig_width = new_size[0] / 30  # 根据图像宽度调整fig宽度
    fig_height = (new_size[1] / 30 )  # 根据色块数量调整fig高度

    # 创建一个figure来显示图像和颜色主题
    fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height))


    # 显示原始图像
    ax[0].imshow(image)
    ax[0].axis('off')
    # ax[0].set_title('Original Image')

    # 设置颜色主题显示
    num_colors = len(colors)
    palette_width = new_size[0]
    palette = np.zeros((50, palette_width, 3), dtype=int)  # 宽度与图像相同
    block_widths = [palette_width // num_colors] * num_colors

    # 确保分配给每个色块的宽度之和等于调色板的总宽度
    for i in range(palette_width % num_colors):
        block_widths[i] += 1

    start_x = 0
    for i, color in enumerate(colors):
        end_x = start_x + block_widths[i]
        palette[:, start_x:end_x] = color
        start_x = end_x

    ax[1].imshow(palette)
    ax[1].axis('off')
    # ax[1].set_title('Color Palette')

    # 在每个颜色块下方显示RGB值
    start_x = 0
    for i, color in enumerate(colors):
        block_center = start_x + block_widths[i] / 2
        rgb_text = f'RGB: ({color[0]}, {color[1]}, {color[2]})'
        hex_text = f'Hex: #{color[0]:02X}{color[1]:02X}{color[2]:02X}'
        final_text = f'{hex_text}\n{rgb_text}'
        
        # 设置文本颜色与色块颜色相同
        text_color = (color[0] / 255, color[1] / 255, color[2] / 255)
        
        # 根据色块宽度调整字体大小
        font_size = max(8, block_widths[i] // 10)  # 确保字体大小不会太小

        # 使字体更明显：增加字体大小和加粗
        ax[1].text(block_center, 60, final_text, fontsize=font_size, ha='center', va='top', color=text_color, fontweight='bold')
        start_x += block_widths[i]

    plt.tight_layout()
    plt.show()

def save_rgb_colors_to_yaml(file_path, sorted_colors):
    # 分解 RGB 颜色列表为三个单独的列表
    r_values = [color[0] for color in sorted_colors]
    g_values = [color[1] for color in sorted_colors]
    b_values = [color[2] for color in sorted_colors]

    # 打开文件并逐行写入
    with open(file_path, 'w') as file:
        file.write("colorR: [")
        file.write( ", ".join(map(str, r_values)) + "]" +"\n")
        file.write("colorG: [")
        file.write( ", ".join(map(str, g_values)) + "]" +"\n")
        file.write("colorB: [")
        file.write( ", ".join(map(str, b_values)) + "]" +"\n")

    # # 构造颜色数据字典
    # color_data_R = {
    #     'colorR': r_values,
    # }
    # color_data_G = {
    #     'colorG': g_values,
    # }
    # color_data_B = {
    #     'colorB': b_values,
    # }
    # # 将数据写入 YAML 文件
    # with open(file_path, 'w') as file:
    #     yaml.dump(color_data_R, file, default_flow_style=True, sort_keys=False)
    #     yaml.dump(color_data_G, file, default_flow_style=True, sort_keys=False)
    #     yaml.dump(color_data_B, file, default_flow_style=True, sort_keys=False)


# 替换为你的图像路径
image_folder = 'background' # album、background、movie
image_name = 'wall7.png'
color_folder = 'colors'
yaml_name = 'wall7.yaml'

# 获取当前脚本的目录
current_directory = os.path.dirname(__file__)
# 获取上一级目录
parent_directory = os.path.dirname(current_directory)
# 构造图片的相对路径
image_path = os.path.join(parent_directory, image_folder, image_name)
output_yaml_path = os.path.join(parent_directory, color_folder, yaml_name)  # 输出的YAML文件路径

num_colors = 8  # 提取的颜色数量
resize_factor = 0.2  # 缩放比例
colors = extract_color_palette(image_path, num_colors, resize_factor)
sorted_colors = sort_colors_by_hue(colors)
save_rgb_colors_to_yaml(output_yaml_path, sorted_colors)
plot_image_and_palette(image_path, sorted_colors)
