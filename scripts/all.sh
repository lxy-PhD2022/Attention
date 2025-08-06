#!/bin/bash

# 定义一个数组，包含所有要运行的脚本文件名
scripts=("etth1.sh" "etth2.sh" "ettm1.sh" "ettm2.sh" "weather.sh" "electricity.sh" "traffic.sh") # 替换为实际的脚本文件名
#   

# 遍历数组中的每个脚本文件
for script in "${scripts[@]}"
do
    echo "Running $script..."
    # 确保脚本文件具有执行权限
    chmod +x scripts/"$script"
    # 执行脚本
    sh scripts/"$script"
    echo "Finished $script"
done

echo "All scripts executed."