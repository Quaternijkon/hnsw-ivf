'''
Author: superestos superestos@gmail.com
Date: 2025-07-09 02:36:33
LastEditors: superestos superestos@gmail.com
LastEditTime: 2025-07-09 02:36:36
FilePath: /dry/faiss/tutorial/python/check_env.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# file: check_faiss_env.py
import faiss
import sys

print("\n" + "="*50)
print("--- Faiss 环境诊断报告 ---")
try:
    # 打印版本号
    print(f"报告的版本号 (faiss.__version__): {faiss.__version__}")

    # 打印 faiss 库的实际文件路径
    print(f"加载的库文件路径 (faiss.__file__): {faiss.__file__}")

    # 检查正在运行的 Python 解释器路径
    print(f"当前 Python 解释器: {sys.executable}")

    # 最终检查: 再次确认 index_io 是否存在
    if hasattr(faiss, 'index_io'):
        print("\n结论: 'faiss.index_io' 模块被成功找到！环境正确。")
    else:
        print("\n结论: 错误！'faiss.index_io' 模块未找到。环境存在问题。")

except Exception as e:
    print(f"\n执行诊断时发生错误: {e}")

print("="*50 + "\n")