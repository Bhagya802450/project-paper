from tkinter import *
import tkinter.ttk as ttk
import csv

root = Tk()
root.title("Crime")
width = 1366
height = 768
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(0, 0)

TableMargin = Frame(root, width=500)
TableMargin.pack(side=TOP)

scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
scrollbary = Scrollbar(TableMargin, orient=VERTICAL)

tree = ttk.Treeview(TableMargin, columns=("src_ip", "dst_ip","protocol","pkt_size","duration","flag_syn"), height=400, selectmode="extended",
                    yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)


scrollbary.config(command=tree.yview)
scrollbary.pack(side=RIGHT, fill=Y)
scrollbarx.config(command=tree.xview)
scrollbarx.pack(side=BOTTOM, fill=X)

tree.heading('src_ip', text="src_ip", anchor=W)
tree.heading('dst_ip', text="dst_ip", anchor=W)
tree.heading('protocol', text="protocol", anchor=W)
tree.heading('pkt_size', text="pkt_size", anchor=W)
tree.heading('duration', text="duration", anchor=W)
tree.heading('flag_syn', text="flag_syn", anchor=W)


tree.column('#0', stretch=NO, minwidth=0, width=0)
tree.column('#1', stretch=NO, minwidth=0, width=120)
tree.column('#2', stretch=NO, minwidth=0, width=120)
tree.column('#3', stretch=NO, minwidth=0, width=120)
tree.column('#4', stretch=NO, minwidth=0, width=120)
tree.column('#5', stretch=NO, minwidth=0, width=120)

tree.pack()




with open('ddos_dataset.csv') as f:
  reader = csv.DictReader(f, delimiter=',')
  for row in reader:
    a1 = row['src_ip']
    a2 = row['dst_ip']
    a3 = row['protocol']
    a4 = row['pkt_size']
    a5 = row['duration']
    a6 = row['flag_syn']



    tree.insert("", 0, values=(a1,a2,a3,a4,a5,a6))
root.mainloop()
