import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "plant_disease_cnn.keras"

IMAGE_SIZE = (128, 128)

CAMERA_INDEX = 0

HEATMAP_THRESHOLD = 0.55

MIN_CONTOUR_AREA = 500


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open("class_names.txt", "r") as file:

    class_names = [
        line.strip()
        for line in file
    ]


print("Classes:", class_names)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, tf.keras.layers.Conv2D):

        last_conv_layer = layer

        break


if last_conv_layer is None:

    raise ValueError(
        "No Conv2D layer found in the trained model."
    )


print(
    "Last Conv2D layer:",
    last_conv_layer.name
)


# ============================================================
# BUILD A NEW FUNCTIONAL GRAPH
#
# IMPORTANT:
# We DON'T use model.output here.
# This avoids the Keras Sequential output error.
# ============================================================

inputs = tf.keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = inputs

conv_output = None


for layer in model.layers:

    x = layer(x)

    # Save output of last Conv2D layer
    if layer == last_conv_layer:

        conv_output = x


# Final prediction
predictions = x


# ============================================================
# GRAD-CAM MODEL
# ============================================================

grad_model = tf.keras.Model(
    inputs=inputs,
    outputs=[
        conv_output,
        predictions
    ]
)


print("Grad-CAM model created successfully!")


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def make_gradcam_heatmap(image):

    """
    Create Grad-CAM heatmap.

    Input:
        image shape = (1, 128, 128, 3)

    Returns:
        heatmap
    """

    with tf.GradientTape() as tape:

        # Get convolution features and prediction
        conv_features, predictions = grad_model(
            image,
            training=False
        )

        # Find predicted class
        predicted_class = tf.argmax(
            predictions[0]
        )

        # Get score for predicted class
        class_score = predictions[
            0,
            predicted_class
        ]


    # ========================================================
    # CALCULATE GRADIENTS
    # ========================================================

    gradients = tape.gradient(
        class_score,
        conv_features
    )


    if gradients is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )


    # ========================================================
    # GLOBAL AVERAGE POOLING
    # ========================================================

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )


    # Remove batch dimension

    conv_features = conv_features[0]


    # ========================================================
    # WEIGHT FEATURE MAPS
    # ========================================================

    heatmap = tf.reduce_sum(
        conv_features * pooled_gradients,
        axis=-1
    )


    # ========================================================
    # RELU
    # ========================================================

    heatmap = tf.maximum(
        heatmap,
        0
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    maximum = tf.reduce_max(
        heatmap
    )


    heatmap = heatmap / (
        maximum +
        tf.keras.backend.epsilon()
    )


    return heatmap.numpy()


# ============================================================
# OPEN WEBCAM
# ============================================================

print()
print("Opening camera...")

camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("==============================================")
print(" REAL-TIME PLANT DISEASE DETECTION")
print("==============================================")
print()
print("Show a plant leaf to the camera.")
print()
print("Heatmap = CNN attention")
print("Red boxes = high activation regions")
print()
print("Press Q to quit.")
print()


# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    success, frame = camera.read()


    if not success:

        print(
            "Could not read camera frame."
        )

        break


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # RESIZE IMAGE
    # --------------------------------------------------------

    resized_image = cv2.resize(
        rgb_frame,
        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # CONVERT TO FLOAT32
    # --------------------------------------------------------

    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # ADD BATCH DIMENSION
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # IMPORTANT:
    #
    # DO NOT use:
    #
    # image_array = image_array / 255.0
    #
    # because your model already contains:
    #
    # Rescaling(1 / 255)
    #
    # as its first layer.


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]


    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )


    predicted_class = class_names[
        predicted_index
    ]


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        float(
            prediction[predicted_index]
        )
        * 100
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    heatmap = make_gradcam_heatmap(
        image_array
    )


    # ========================================================
    # RESIZE HEATMAP TO CAMERA SIZE
    # ========================================================

    heatmap = cv2.resize(
        heatmap,
        (
            frame.shape[1],
            frame.shape[0]
        )
    )


    # ========================================================
    # CREATE HEATMAP
    # ========================================================

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )


    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )


    # ========================================================
    # OVERLAY HEATMAP
    # ========================================================

    output_frame = cv2.addWeighted(
        frame,
        0.60,
        colored_heatmap,
        0.40,
        0
    )


    # ========================================================
    # CREATE MASK
    # ========================================================

    mask = (
        heatmap >= HEATMAP_THRESHOLD
    ).astype(
        np.uint8
    ) * 255


    # ========================================================
    # CLEAN MASK
    # ========================================================

    kernel = np.ones(
        (7, 7),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # FIND HIGH-ACTIVATION REGIONS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # DRAW REGIONS
    # ========================================================

    region_count = 0


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # Ignore tiny areas

        if area < MIN_CONTOUR_AREA:

            continue


        region_count += 1


        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )


        # ----------------------------------------------------
        # RED BOX
        # ----------------------------------------------------

        cv2.rectangle(
            output_frame,
            (x, y),
            (
                x + width,
                y + height
            ),
            (0, 0, 255),
            3
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        cv2.putText(
            output_frame,
            "CNN attention",
            (
                x,
                max(y - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        output_frame,
        (0, 0),
        (520, 115),
        (0, 0, 0),
        -1
    )


    # --------------------------------------------------------
    # DISEASE
    # --------------------------------------------------------

    cv2.putText(
        output_frame,
        f"Prediction: {predicted_class}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    cv2.putText(
        output_frame,
        f"Confidence: {confidence:.2f}%",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # REGION COUNT
    # --------------------------------------------------------

    cv2.putText(
        output_frame,
        f"Highlighted regions: {region_count}",
        (15, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (0, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Plant Disease - CNN + Grad-CAM",
        output_frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLOSE CAMERA
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("Camera closed.")