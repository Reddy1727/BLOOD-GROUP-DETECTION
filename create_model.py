import os
os.chdir(r"C:\Users\Reddy Y L\Documents\Projects\fingerprint-based-blood-group-detection-main")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Create a simple model
model = Sequential([
    Flatten(input_shape=(256, 256, 3)),
    Dense(128, activation='relu'),
    Dense(8, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Ensure directory exists
os.makedirs('test', exist_ok=True)

# Save the model
model.save('test/model_blood_group_detection_resnet.h5')
print(f"✓ Model saved to: test/model_blood_group_detection_resnet.h5")
print(f"✓ File exists: {os.path.exists('test/model_blood_group_detection_resnet.h5')}")
