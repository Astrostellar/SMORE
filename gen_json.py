import pandas as pd 
import json
import glob
from sklearn.model_selection import train_test_split

root="/hpc/data/home/bme/v-wangxin/szr_code/NILM_images_train_SR/"
paths=glob.glob(f'{root}/*.jpg')
labels=[0]*len(paths)

train_X,test_X,train_y,test_y = train_test_split(paths,labels,test_size=0.2)
data={}
data['train']=train_X
data['val']=test_X

f=open('datasets/NILM.json','w')
json.dump(data,f)
f.close()