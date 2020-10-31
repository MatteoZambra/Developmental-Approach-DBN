import os
path = os.getcwd()
files = os.listdir(path)


for index, file in enumerate(files):
    if file[-4:] == '.png':
        print(file)
        # print( os.path.join(path, 'mnist_{}'.format(file)) )
        os.rename(os.path.join(path, file), os.path.join(path, 'mnist_{}'.format(file)) )