from alghoritms import *
from queue import Queue

tree = BinTree()
node = BinNode(key=3)
node1 = BinNode(key=4)
node2 = BinNode(key=2)
node3 = BinNode(key=1)
tree.insert(node)
tree.insert(node1)
tree.insert(node2)
tree.insert(node3)
print(tree)
print()
# tree.delete(node)
print(tree.search(3))