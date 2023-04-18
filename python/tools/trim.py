import tkinter as tk

def remove_newlines():
    input_text = text_input.get("1.0", tk.END)
    output_text = input_text.replace('\n', ' ')
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, output_text)

# 创建主窗口
root = tk.Tk()
root.title("Remove Newlines")

# 创建输入文本框
text_input = tk.Text(root, wrap=tk.WORD, height=10, width=50)
text_input.pack(padx=5, pady=5)

# 创建处理按钮
process_button = tk.Button(root, text="Remove Newlines", command=remove_newlines)
process_button.pack(padx=5, pady=5)

# 创建输出文本框
text_output = tk.Text(root, wrap=tk.WORD, height=10, width=50)
text_output.pack(padx=5, pady=5)

# 启动主事件循环
root.mainloop()
