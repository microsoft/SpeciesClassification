
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


cm = np.loadtxt('./conf_val_unnormalized.csv', delimiter=',')
# rows are ground-truth class and columns  the predicted class
# see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html for more details
classnames = np.loadtxt('./classlist.txt', dtype=str, delimiter=',')


# In[3]:


train_image_count = np.loadtxt('./class_count_trainval.txt', dtype=int)[:,1]


# In[4]:


cm_normalized = cm / cm.sum(axis=0)


# The entry in a row, that is on the diagonal, are true positive, all other entries are false negative.
# The entry in a column, that is on the diagonal, are true positive, all other entries are false positive

# In[5]:


predictions_per_class = np.sum(cm, axis = 0)
samples_per_class = np.sum(cm, axis = 1)


# In[6]:


precision = np.diag(cm) / predictions_per_class
recall = np.diag(cm) / samples_per_class


# In[8]:


np.savetxt('precision_per_class.txt', precision, fmt='%.6f')
np.savetxt('recall_per_class.txt', recall, fmt='%.6f')


# In[63]:


print('{} classes are never predicted'.format(np.sum(np.isclose(0, predictions_per_class))))


# In[64]:


print('{} classes have a recall of 0.5 or less'.format(np.sum(recall< 0.5)))


# In[95]:


plt.scatter(train_image_count, recall, s=1)
plt.xlabel('Number of class samples in trainval data')
plt.ylabel('Recall of that class')
plt.xlim([-5,1500])
plt.savefig('trainimages_vs_recall.png', format='png')


# In[97]:


plt.figure()
plt.scatter(train_image_count, predictions_per_class, s=1)
plt.xlabel('Number of class samples in trainval data')
plt.ylabel('Number of times predicted')
plt.xlim([-5,1500])
plt.savefig('trainimages_vs_numberpredictions.png', format='png')


# In[46]:


np.fill_diagonal(cm_normalized, 0)


# In[49]:


largest_confusions = np.argsort(-cm_normalized.ravel())


# In[81]:


for ind in largest_confusions[:10]:
    first_c, second_c = np.unravel_index(ind, cm_normalized.shape)
    print('Confused {:.2%} of images of class {} with the class {}'.format(
                cm_normalized.ravel()[ind],
                classnames[first_c], 
                classnames[second_c]))

