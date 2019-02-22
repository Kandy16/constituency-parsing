import nltk as nltk

## https://www.nltk.org/_modules/nltk/tree.html
## http://www.nltk.org/howto/tree.html
## above link contains the API and also some tutorials

dirname = os.getcwd()
dirname = os.path.dirname(dirname)
dataset_path = os.path.join(dirname, 'datasets/')
print(dataset_path)

#reader = nltk.corpus.BracketParseCorpusReader('.','SWB-all-sentences-original-with-punctuation.MRG')
reader = nltk.corpus.BracketParseCorpusReader(dataset_path,'WSJ.txt')
print(reader.fileids())
print(type(reader))

## reads the file and converts each line into a tree
trees = reader.parsed_sents()
print('No. of trees: ', len(trees))
tree = trees[2] #get the 3rd tree in the examples (it looks reasonable)

print(tree)
print('Type of tree : ',type(tree))

print('Label : ',tree.label()) # header of the tree - mostly the POS is stored
print('Leaves : ', tree.leaves()) # gives out all the leaves in string format

print('Children of parent node :', len(tree))

# how to traverse the tree
# the children can be accessed through indexing

subTree = tree[0] #
print(type(subTree))
print(len(subTree))
print(subTree.label())

# the immediate children from left to right can be accessed this way
for i in range(len(subTree)):
    print(subTree[i].label())
    print(subTree[i])

print(subTree[2].label())
print(len(subTree[2]))

# depth of the tree can be achieved through this way
print('Tree Height:', tree.height())
print(tree.treepositions())
print(tree.treepositions('leaves'))

# all the trees are obtained in this fashion
subTrees = tree.subtrees()
print(type(subTrees))
for sub in subTrees:
    print(sub)


# traverse a tree through depth-first search
def traverse_depthFirst(tree):
    if(type(tree[0]) == type('a string')):
        print(tree.label() + ' : ' + tree[0])
        print('depth is reached !!!')
        return

    for i in range(len(tree)):
        print('Inside tree : '+ tree[i].label())
        traverse_depthFirst(tree[i])


print(traverse_depthFirst(tree))


t = nltk.tree.Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
print(t)
traverse_depthFirst(t)

import numpy as np

def build_tree_randomly(entity):
    if(type(entity) == type('a string')):
        token_list = entity.split()
        leaf_list = [nltk.tree.Tree(token, [token]) for token in token_list]
        return build_tree_randomly(leaf_list)

    # Get the length of list. Run a random generator to select an item
    # Combine the consecutive elements. Do this recursively
    if(len(entity) > 1):
        tree_list = entity
        chosen_entity_no = np.random.randint(0, len(tree_list) - 1) # choose between 1st and second last item
        new_tree_label = tree_list[chosen_entity_no].label() + '+' + tree_list[chosen_entity_no + 1].label()
        new_tree = nltk.tree.Tree(new_tree_label, [tree_list[chosen_entity_no], tree_list[chosen_entity_no + 1]])

        tree_list[chosen_entity_no:chosen_entity_no+2] = [new_tree]
        return build_tree_randomly(tree_list)

    else:
        return entity[0]


output_tree = build_tree_randomly('I am a string')
print(output_tree)