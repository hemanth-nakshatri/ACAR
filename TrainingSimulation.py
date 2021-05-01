print("SETTING UP")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils import *
from sklearn.model_selection import train_test_split


# STEP 1 --> Import Data
path = 'MyData'
data = importDataInfo(path)


# STEP 2  --> Visualize and Balance Data
data = balanceData(data,display=True)


# STEP 3  --> Prepare for Pre-Processing
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0], steerings[0])


# STEP 4 --> Split Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print('Total train images: ', len(X_train))
print('Total test images: ', len(X_test))


# STEP 5 --> Data Augmentation

# STEP 6 --> Pre-processing

# STEP 7  --> Batch Generation

# STEP 8  --> Creating CNN model
model = createModel()
model.summary()

# STEP 9  --> Train Model
history = model.fit(batchGen(X_train, y_train, 100, 1), steps_per_epoch=300, epochs=25,
          validation_data=batchGen(X_test,y_test,50,0), validation_steps=200, verbose=1)

# STEP 10  --> Save model and PLOT
model.save('models\model_phase_1.2.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Training", "Validation"])
plt.title('Loss')
plt.xlabel('Epoch')

plt.show()