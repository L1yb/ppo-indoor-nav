#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
f-string转换工具
用于将Python 3.6+的f-strings转换为format()方法，适配老版本Python
"""

import re
import os
import sys
import argparse

def find_fstrings(line):
    """查找行中的f-strings"""
    # 简单的f-string匹配模式
    pattern = r'f"(.*?)"'
    alt_pattern = r"f'(.*?)'"
    
    matches = []
    for p in [pattern, alt_pattern]:
        for match in re.finditer(p, line):
            matches.append((match.start(), match.end(), match.group(1)))
    
    return sorted(matches, key=lambda x: x[0])

def process_fstring_content(content):
    """处理f-string内容，提取表达式和格式说明符"""
    expressions = []
    format_parts = []
    
    # 匹配{expr}或{expr:format}
    pattern = r'\{([^{}:]+)(?::([^{}]+))?\}'
    
    # 将{表达式}替换为{}，并收集表达式
    processed = re.sub(pattern, lambda m: process_match(m, expressions, format_parts), content)
    
    return processed, expressions, format_parts

def process_match(match, expressions, format_parts):
    """处理单个表达式匹配"""
    expr = match.group(1).strip()
    format_spec = match.group(2)
    
    expressions.append(expr)
    if format_spec:
        format_parts.append(f":{format_spec}")
    else:
        format_parts.append("")
    
    return "{}"

def convert_fstring_to_format(line):
    """将f-string转换为format()方法"""
    fstrings = find_fstrings(line)
    if not fstrings:
        return line
    
    # 从后向前替换，避免位置偏移
    result = line
    for start, end, content in reversed(fstrings):
        processed, expressions, format_parts = process_fstring_content(content)
        
        # 构建format表达式
        if expressions:
            format_args = ", ".join(expressions)
            # 检查是否需要添加格式说明符
            for i, fmt in enumerate(format_parts):
                if fmt:
                    processed = processed.replace("{}", "{" + fmt + "}", 1)
                else:
                    processed = processed.replace("{}", "{}", 1)
            
            format_str = f'"{processed}".format({format_args})'
            result = result[:start] + format_str + result[end:]
        else:
            # 没有表达式的情况，只是一个普通字符串
            format_str = f'"{content}"'
            result = result[:start] + format_str + result[end:]
    
    return result

def process_file(filepath, backup=True):
    """处理单个文件，转换所有f-strings"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as file:
                lines = file.readlines()
        except Exception as e:
            print(f"无法读取文件 {filepath}: {e}")
            return False
    
    # 创建备份
    if backup:
        backup_path = f"{filepath}.bak"
        try:
            with open(backup_path, 'w', encoding='utf-8') as bak:
                bak.writelines(lines)
        except Exception as e:
            print(f"无法创建备份文件 {backup_path}: {e}")
            return False
    
    # 转换f-strings
    modified = False
    converted_lines = []
    for line in lines:
        converted = convert_fstring_to_format(line)
        if converted != line:
            modified = True
        converted_lines.append(converted)
    
    # 如果有修改，写入文件
    if modified:
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.writelines(converted_lines)
            print(f"已转换 {filepath}")
        except Exception as e:
            print(f"写入文件 {filepath} 时出错: {e}")
            return False
    else:
        print(f"文件 {filepath} 中没有f-strings需要转换")
    
    return True

def walk_directory(directory, extensions=None, recursive=True):
    """遍历目录中的Python文件"""
    if extensions is None:
        extensions = ['.py']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                yield os.path.join(root, file)
        
        if not recursive:
            break

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将Python 3.6+的f-strings转换为format()方法')
    parser.add_argument('path', help='要处理的文件或目录')
    parser.add_argument('--no-backup', action='store_true', help='不创建备份文件')
    parser.add_argument('--no-recursive', action='store_true', help='不递归处理子目录')
    parser.add_argument('--ext', action='append', default=['.py'], help='要处理的文件扩展名')
    
    args = parser.parse_args()
    
    path = args.path
    create_backup = not args.no_backup
    recursive = not args.no_recursive
    extensions = args.ext
    
    if os.path.isfile(path):
        success = process_file(path, create_backup)
        return 0 if success else 1
    elif os.path.isdir(path):
        success = True
        for filepath in walk_directory(path, extensions, recursive):
            if not process_file(filepath, create_backup):
                success = False
        return 0 if success else 1
    else:
        print(f"错误: 路径 {path} 不存在")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 