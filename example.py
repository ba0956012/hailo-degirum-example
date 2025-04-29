import degirum as dg
import degirum_tools
import cv2
from picamera2 import Picamera2
import traceback

# zoo_url = "degirum/hailo"
zoo_url = "./model/scrfd.json"
token = ""  # Optional: fill in if needed
device_type = "HAILORT/HAILO8L"

inference_host_address = "@local"

# Choose the model name
# face_det_model_name = "scrfd_10g--640x640_quant_hailort_hailo8l_1"
face_det_model_name = "scrfd"


# Load AI model
model = dg.load_model(
    model_name=face_det_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url,
    token=token,
    device_type=device_type,
)


face_rec_model_name = "arcface_mobilefacenet"
face_rec_model_name = "arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1"
# Load the face recognition model
face_rec_model = dg.load_model(
    model_name=face_rec_model_name,
    inference_host_address="@local",
    # zoo_url="degirum/models_hailort"
    zoo_url = "./model/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1/arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1.json"
)
print(dir(face_rec_model))



def display_images(images, title="Images", figsize=(15, 5)):
    """
    Display a list of images in a single row using Matplotlib.

    Parameters:
    - images (list): List of images (NumPy arrays) to display.
    - title (str): Title for the plot.
    - figsize (tuple): Size of the figure.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]  # Make it iterable for a single image
    for ax, image in zip(axes, images):
        image_rgb = image[:, :, ::-1]  # Convert BGR to RGB
        ax.imshow(image_rgb)
        ax.axis('off')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

picam2 = Picamera2()
picam2.start()

# Setup for Display
with degirum_tools.Display("AI Camera") as output_display:
    while True:
        # Capture frame-by-frame from the camera
        # ret, frame = cap.read()
        frame = picam2.capture_array()

        # if not ret:
        #     print("Error: Failed to capture image.")
        #     break

        # Convert the frame to RGB (if needed for your model)
        # Optional: Use a different color format if your model expects something else
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = rgb_frame

        # Run inference on the captured frame
        inference_result = model(rgb_frame)
        """
        print(type(inference_result))
        print(dir(inference_result))
        print(type(inference_result.results))
        print(len(inference_result.results))
        
        for result in inference_result.results:
            
            print(type(inference_result.results))
        
        for name, tensor in inference_result.results[0].items():
            print(name)
            print(tensor)
        for info in inference_result.results[0]["data"][0][:5]:
            print(info)
        
        break
        """
        cropped_faces = []
        for face in inference_result.results:
            x1, y1, x2, y2 = map(int, face["bbox"])
            cropped_face = rgb_frame[y1:y2, x1:x2]
            cropped_faces.append(cropped_face) 
        
        # print(cropped_faces)
        if cropped_faces:
            face_embedding = face_rec_model(cropped_faces[0]).results[0]["data"][0]
            # print(face_embedding)
            cv2.imshow("Face Detection", cropped_faces[0])
        else:
        
            # Display the image overlay with the face detection results
            # output_display.show_image(inference_result.image_overlay)

            # Display the frame in a window (optional, for debugging)
            cv2.imshow("Face Detection", inference_result.image_overlay)

        # Exit loop if 'x' or 'q' is pressed
        if cv2.waitKey(1) & 0xFF in [ord("x"), ord("q")]:
            print("Exiting...")
            break

    # Release the camera and close any OpenCV windows
    # cap.release()
    cv2.destroyAllWindows()
