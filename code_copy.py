'''
create by xubing on 2020-03-17

一个简陋的拷贝代码到word里的程序
直接调用code_copy(base_dir, out_docx)
base_dir表示代码所在的文件目录（是目录）
out_docx表示输出的word文件（是文件）


'''

import os

from docx import Document


def code_copy(base_dir, out_docx):
    """
    拷贝代码到word文档里
    :param base_dir: 要拷贝的源程序文件夹
    :param out_docx: 要输出的docx文件
    :return: 无
    """
    document = Document()
    file_list = os.listdir(base_dir)
    for file in file_list:
        # 排除以下三种情况
            ## file不能是文件夹
            ## out_docx不在base_dir文件夹下面
            ## 排除macOS中的.DS_Store文件

        if not os.path.isdir(base_dir + file) and base_dir + file != out_docx and file != '.DS_Store':
            with open(base_dir + file, 'r', encoding='utf8') as f:
                print(base_dir + file)
                content = f.read()
                document.add_paragraph(content)
    document.save(out_docx)


if __name__ == '__main__':
    # base_dir1 = '/Users/xubing/Projects/mirror/mirror_web/src/views/model/components/'
    # out_docx1 = '/Users/xubing/Desktop/demo1.docx'
    
    base_dir2 = '/Users/xubing/Projects/mirror/mirror_inference_web/src/styles/'
    out_docx2 = '/Users/xubing/Desktop/demo2.docx'

    base_dir3 = '/Users/xubing/Projects/mirror/mirror_inference_web/'
    out_docx3 = '/Users/xubing/Desktop/demo3.docx'

    code_copy(base_dir3, out_docx3)
