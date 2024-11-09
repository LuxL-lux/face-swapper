
from face_swapper.model import FaceSwapper
from face_swapper.config import get_images_dir
from pathlib import Path
from PIL import Image
import cv2
import av

def show_webcam(mirror=False):
    swapper = FaceSwapper(checkpoints_dir=Path("C:/Users/lukas.lux/RemoteIngferencePackages/fastapi-aiortc/data/checkpoints"),preferred_provider='cpu')
    source_images = [Image.open(Path("C:/Users/lukas.lux/RemoteIngferencePackages/fastapi-aiortc/data/images/example_person.jpg"))]
    container = av.open(file='0', format='vfwcap')
    stream = container.streams.video[0]
    
    import time
    last_process_time = time.time()
    min_frame_interval = 1.0/30.0  # Cap at 30 FPS
    frame_interval = min_frame_interval
    fps = 0
    fps_update_interval = 0.5
    last_fps_update = time.time()
 
    for frame in container.decode(stream):
        current_time = time.time()
        
        if current_time - last_process_time < frame_interval:
            continue
            
        if frame is not None:
            process_start = time.time()
            
            np_arr = frame.to_ndarray(format='bgr24')
            # processed_frame = swapper.process_frame(np_arr, source_images)
            processed_frame = np_arr
            if current_time - last_fps_update > fps_update_interval:
                fps = 1.0 / frame_interval
                last_fps_update = current_time
            
            cv2.putText(processed_frame, 
                       f'FPS: {fps:.1f}', 
                       (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)
            
            cv2.imshow('FaceSwap Live', processed_frame)
            
            process_duration = time.time() - process_start
            frame_interval = max(process_duration * 1.2, min_frame_interval)
            last_process_time = current_time
        
        if cv2.waitKey(1) == 27:
            break
            
    container.close()
    cv2.destroyAllWindows()
def inference():
    swapper = FaceSwapper("./checkpoints/models/inswapper_128.onnx", preferred_provider='openvino')
    source_images = [Image.open("./data/face_to_swap2.jpg")]
    target_image = Image.open("./data/target_face_2.jpeg")
    result = swapper.process(source_images, target_image, source_indexes=-1, target_indexes=-1)
    result.save("result.png")

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()
