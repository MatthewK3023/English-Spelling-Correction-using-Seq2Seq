import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

score = np.load('weights/testing_bleu4.npy')
loss = np.load('weights/training_loss.npy')

plt.figure(figsize=(15,15))
plt.subplot(2,1,1)
sns.set_style("whitegrid")
plt.plot(loss)
#plt.title('Training loss curve', fontsize=13)
plt.xlabel('Number of iterations', fontsize=13)
plt.ylabel('Training loss', fontsize=13)

plt.subplot(2,1,2)
plt.plot(score)
#plt.title('BLUE-4 score testing curve during training', fontsize=13)
plt.xlabel('Number of iterations', fontsize=13)
plt.ylabel('BLUE-4 score', fontsize=13)
plt.savefig(f'training_process/training_process.png')
plt.show()