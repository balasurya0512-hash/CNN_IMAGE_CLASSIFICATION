import tensorflow as tf
import numpy as np


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    r"C:\Users\sarath kumar b\Desktop\New folder (4)\plant_disease_cnn.keras"
)


# ==========================================
# 2. LOAD CLASS NAMES
# ==========================================

with open(r"C:\Users\sarath kumar b\Desktop\New folder (4)\class_names.txt", "r") as file:

    class_names = [
        line.strip()
        for line in file
    ]


print("Classes:", class_names)


# ==========================================
# 3. IMAGE PATH
# ==========================================

image_path = r"C:\Users\sarath kumar b\Desktop\New folder (4)\Computer vision\Powdery\8d5d348133579b53.jpg"


# ==========================================
# 4. LOAD IMAGE
# ==========================================

image = tf.keras.utils.load_img(
    image_path,
    target_size=(128, 128)
)


# ==========================================
# 5. CONVERT TO ARRAY
# ==========================================

image_array = tf.keras.utils.img_to_array(
    image
)


# ==========================================
# 6. ADD BATCH DIMENSION
# ==========================================

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ==========================================
# 7. PREDICT
# ==========================================

prediction = model.predict(
    image_array,
    verbose=0
)[0]


# ==========================================
# 8. FIND HIGHEST PROBABILITY
# ==========================================

predicted_index = np.argmax(
    prediction
)

predicted_class = class_names[
    predicted_index
]

confidence = prediction[
    predicted_index
] * 100


# ==========================================
# 9. RESULT
# ==========================================

print()
print("==============================")
print("PLANT DISEASE PREDICTION")
print("==============================")

print(
    f"Disease: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)